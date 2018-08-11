"""
    功能: 棋谱训练tensorflow实现
    initialize_weight(var_name, var_shape): 使用Xavier方法初始化权重
    initialize_bias(bia_name, bia_shape): 初始化偏置
    TFProcess类: 训练网络的构造，训练权重的保存
"""
from pathlib import Path

from common import *


def initialize_weight(var_name, var_shape):
    """
        功能: 使用Xavier方法初始化权重
        输入:
            var_name: 卷积核的tensor名
            var_shape: 卷积核的tensor形状
        返回:
            weights_var: 卷积核tensor
    """
    weights_var = tf.get_variable(var_name,
                                  initializer=tf.truncated_normal(var_shape,
                                                                  np.sqrt(2.0 / (sum(var_shape)))))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights_var)
    return weights_var


def initialize_bias(bia_name, bia_shape):
    """
        功能: 初始化偏置
        输入:
            bia_name: 偏置的tensor名
            bia_shape: 偏置的tensor形状
        返回:
            bias_var: 偏置tensor
    """
    bias_var = tf.get_variable(bia_name,
                               initializer=tf.constant(0.0, shape=bia_shape))
    return bias_var


class XTSETensorFlow:
    initial_lr = 0.03  # 初始学习率
    max_step = 10000  # 最大训练步数
    train_swa = 1  # 是否保存权重的滑动平均. 1：保存；0：不保存
    train_swa_step = 100  # 对权重做加权平均的步数

    def __init__(self, learning_rate, max_step, swa, swa_step):
        """
            功能: 训练参数初始化
            输入:
                learning_rate: 初始学习率
                max_step: 最大训练步数
                swa: 是否保存权重的滑动平均
                swa_step: 对权重做滑动平均的步数
            返回:
                无返回值
        """
        self.FILTER_CHANNELS = 256  # 卷积通道数
        self.XT_SE_BLOCK_NUM = 20  # xtResNet_SE的模块数量
        self.cardinality = 2  # xtResNet的分支数量
        self.reduction_ratio = 4  # SE的全连接部分输入-输入神经元个数的比例

        self.initial_lr = learning_rate  # 初始学习率
        self.max_step = max_step  # 最大训练步数
        self.train_swa = swa  # 是否保存权重的滑动平均. 1：保存；0：不保存
        self.train_swa_step = swa_step  # 对权重做滑动平均的步数

        print("The initialized train parameters include LEARNING RATE: {0}, MAX STEP:{1}, SWA:{2} and SWA_STEP:{3}" \
              .format(self.initial_lr, self.max_step, self.train_swa, self.train_swa_step))

        self.gpus_num = GPU_NUM  # 用于训练的GPU数量
        self.weights = []  # 权重列表,用于保存计算图的训练权重

        #  process函数会用到
        self.avg_policy_loss = []
        self.avg_mse_loss = []
        self.avg_reg_term = []
        self.time_start = None

        self.swa_enabled = (self.train_swa == 1)  # 是否保存滑动平均权重
        self.swa_c = 1
        self.swa_max_n = 16
        self.swa_recalc_bn = True

        config = tf.ConfigProto(allow_soft_placement=True)  # GPU资源消耗动态增加，避免内存资源不够
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def init_handle(self, data_set, train_iterator, test_iterator):
        """
            功能: 初始化句柄
            输入:
                dataset: 数据集
                train_iterator: 训练迭代器
                test_iterator: 测试迭代器
            返回:
                无返回值
        """
        print("init_handle")
        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(self.handle, data_set.output_types, data_set.output_shapes)
        self.next_batch = iterator.get_next()
        self.train_handle = self.session.run(train_iterator.string_handle())
        self.test_handle = self.session.run(test_iterator.string_handle())
        self.initialize_network(self.next_batch, self.gpus_num)
        print("finish init_handle")

    def split_data_batch(self, data_batch, gpu_num):
        """
            功能: 将数据划分到每个GPU上
            输入:
                data_batch: 数据集批次
                gpu_num: GPU数量，默认为1
            返回:
                sx: 棋谱tensor
                sy: 走子概率tensor
                sz: 胜负概率tensor
        """
        sx = tf.split(data_batch[0], gpu_num)  # 将棋谱tensor平均分配到每个GPU上
        sy = tf.split(data_batch[1], gpu_num)  # 将走子概率tensor平均分配到每个GPU上
        sz = tf.split(data_batch[2], gpu_num)  # 将胜负概率tensor平均分配到每个GPU上

        return sx, sy, sz

    def initialize_network(self, next_batch, gpus_num=1):
        """
            功能: 初始化训练网络，将数据划分到每个GPU上，汇总各个GPU上得到的损失值并计算平均值
            输入:
                next_batch: 数据集批次
                gpus_num: GPU数量，默认为1
            返回:
                无返回值
        """
        print("initialize_network")
        self.y_ = next_batch[1]
        self.sx, self.sy_, self.sz_ = self.split_data_batch(next_batch, gpus_num)
        self.bn_num = 0  # 对bn操作的次数进行计数
        self.variable_reused = None  # 设置tf的变量是否可重用，关系到运行是否能正常执行

        # 反向传播时的下降方法
        opt_op = tf.train.MomentumOptimizer(learning_rate=self.initial_lr, momentum=0.9, use_nesterov=True)

        tower_grads = []  # 保存每个GPU上的变量的梯度
        tower_loss, tower_policy_loss, tower_mse_loss = [], [], []  # 保存每个GPU上的训练损失, 策略网络损失, 平方差损失,
        tower_reg_term = []  # 保存每个GPU上的正则化项
        tower_y_conv = []  # 保存每个GPU上的策略头得分

        counter = 0  # 计数器

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(gpus_num):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("tower_%d" % i):
                        print("gpu_" + str(i) + "_tower_" + str(i))
                        loss, policy_loss, mse_loss, reg_term, y_conv = self.tower_loss(
                            self.sx[counter], self.sy_[counter], self.sz_[counter])
                        counter += 1

                        self.reset_bn_num()  # 每个GPU计算各自的bn操作数

                        tf.get_variable_scope().reuse_variables()
                        grads = opt_op.compute_gradients(loss)  # 梯度下降

                        tower_grads.append(grads)
                        tower_loss.append(loss)
                        tower_policy_loss.append(policy_loss)
                        tower_mse_loss.append(mse_loss)
                        tower_reg_term.append(reg_term)
                        tower_y_conv.append(y_conv)

        self.mean_grads = self.average_gradients(tower_grads)  # 计算对网络权重的平均梯度
        self.loss = tf.reduce_mean(tower_loss)  # 计算平均总损失
        self.policy_loss = tf.reduce_mean(tower_policy_loss)  # 计算策略网络的平均损失
        self.mse_loss = tf.reduce_mean(tower_mse_loss)  # 计算估值网络的平均损失
        self.reg_term = tf.reduce_mean(tower_reg_term)  # 计算正则化项的平均值
        self.y_conv = tf.concat(tower_y_conv, axis=0)  # 计算策略网络的平均得分

        if self.swa_enabled is True:  # 是否做加权平均
            self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
            self.swa_skip = tf.Variable(self.swa_c, name='swa_skip', trainable=False)
            accum = []  # 加权平均后的权重列表1
            load = []  # 加权平均后的权重列表2
            n = self.swa_count  # 初始为0
            for w in self.weights:
                var = tf.Variable(tf.zeros(shape=w.shape), name='swa/' + w.name.split(':')[0], trainable=False)
                accum.append(tf.assign(var, var * (n / (n + 1.)) + w * (1. / (n + 1.))))
                load.append(tf.assign(w, var))

            with tf.control_dependencies(accum):
                self.swa_accum_op = tf.assign_add(n, 1.)  # 累加操作
            self.swa_load_op = tf.group(*load)  # 转换为tf操作

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 获取计算图的更新操作
        with tf.control_dependencies(self.update_ops):  # tf控制依赖
            self.train_op = opt_op.apply_gradients(self.mean_grads, global_step=self.global_step)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_conv, 1),
                                                        tf.argmax(self.y_, 1)),
                                               tf.float32))

        # 训练日志保存
        self.test_log = tf.summary.FileWriter(get_path("xtResNetSElogs/test"), self.session.graph)
        self.train_log = tf.summary.FileWriter(get_path("xtResNetSElogs/train"), self.session.graph)

        self.init_var = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.session.run(self.init_var)

        print("finish initialize_network")

    def average_gradients(self, tower_grads):
        """
            功能: 将各个GPU上计算得到的梯度汇总求平均值
            输入:
                tower_grads: 来着各个GPU的梯度数据
            返回:
                average_grads: 计算得到的平均梯度
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                grads.append(tf.expand_dims(g, 0))

            average_grads.append((tf.reduce_mean(tf.concat(grads, 0), 0),
                                  grad_and_vars[0][1]))

        return average_grads

    def tower_loss(self, x, y_, z_):
        """
            功能: 计算损失值
            输入:
                x: 棋谱数据
                y_: 棋谱的走子概率标签
                z_: 棋谱的胜率标签
            返回:
                loss: 总的损失值
                policy_loss: 策略网络的输出损失
                mse_loss: 胜率估值网络的输出损失
                reg_term: 正则化后的权重
        """
        print("tower_loss")
        y_conv, z_conv = self.construct_cnn_policy_value_net(x)  # 获取策略头和估值头得分

        policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  # 策略头损失
        mse_loss = tf.reduce_mean(tf.squared_difference(z_, z_conv))  # 估值头损失

        # 正则化减轻过拟合
        reg_term = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale=0.0001),
                                                          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        loss = 1.0 * policy_loss + 1.0 * mse_loss + reg_term  # 计算训练损失

        print("finish tower_loss")

        return loss, policy_loss, mse_loss, reg_term, y_conv

    def reset_bn_num(self):
        """
            功能: 建立棋谱训练网络，包括卷积网络，策略网络和估值网络
            输入:
                planes: 棋谱数据
            返回:
                h_fc1: 策略网络得分
                h_fc3: 策略网络得分
        """
        self.bn_num = 0
        self.variable_reused = True

    def add_weights(self, variable):
        """
            功能: 将权重变量添加到列表中
            输入:
                variable: 权重变量
            返回:
                无
        """
        if self.variable_reused is None:
            self.weights.append(variable)

    def get_bn_num(self):
        """
            功能: 对bn操作的数量计数, 生成bn键值
            输入:
                无
            返回:
                bn键值
        """
        result = "bn" + str(self.bn_num)
        self.bn_num += 1
        return result

    def add_conv_bn_weight(self, conv_var, conv_scope_name):
        """
            功能: 将卷积变量以及对应的BN均值和方差加入列表中
            输入:
                conv_var: 卷积变量
                conv_scope_name: 卷积变量名
            返回:
                无
        """
        beta_key = conv_scope_name + "/batch_normalization/beta:0"
        mean_key = conv_scope_name + "/batch_normalization/moving_mean:0"
        var_key = conv_scope_name + "/batch_normalization/moving_variance:0"

        beta = tf.get_default_graph().get_tensor_by_name(beta_key)
        mean = tf.get_default_graph().get_tensor_by_name(mean_key)
        var = tf.get_default_graph().get_tensor_by_name(var_key)

        self.add_weights(conv_var)
        self.add_weights(beta)
        self.add_weights(mean)
        self.add_weights(var)

    def conv_and_bn_block(self, inputs, filter_size, input_channels, output_channels, name):
        """
            功能: 卷积核BN操作模块，将可训练变量以及BN的均值和方差加入列表中
            输入:
                inputs: 输入特征
                filter_size: 滤波核尺寸
                input_channels: 输入通道
                output_channels: 输出通道
                name: tf变量名
            返回:
                h_bn: 输出特征
        """
        W_conv = initialize_weight(name, [filter_size, filter_size, input_channels, output_channels])  # 初始化卷积核
        weight_key = self.get_bn_num()  # 获取BN编号

        with tf.variable_scope(weight_key):
            input_conv = tf.nn.conv2d(inputs, W_conv, data_format='NCHW',
                                      strides=[1, 1, 1, 1], padding='SAME')
            h_bn = tf.layers.batch_normalization(input_conv, epsilon=1e-5, axis=1,
                                                 fused=True, center=True, scale=False,
                                                 training=self.training, reuse=self.variable_reused)

        self.add_conv_bn_weight(W_conv, weight_key)

        return h_bn

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):  # se网络
        """
            功能: SEnet方法中的se层
            输入:
                input_x: 输入特征
                out_dim: 输出通道数
                ratio: 比例
                layer_name: tf变量名
            返回:
                x: 做SE操作后的特征
        """
        squeeze = tf.reduce_mean(input_x, [2, 3])  # 全局平均池化
        W_fc1 = initialize_weight(layer_name + 'w_fc_1', [out_dim, int(out_dim / ratio)])
        self.add_weights(W_fc1)
        excitation = tf.matmul(squeeze, W_fc1)
        excitation = tf.nn.relu(excitation)
        W_fc2 = initialize_weight(layer_name + 'w_fc_2', [int(out_dim / ratio), out_dim])
        self.add_weights(W_fc2)
        excitation = tf.matmul(excitation, W_fc2)
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1, out_dim, 1, 1])  # -1表示自动计算第0维数据
        scale = input_x * excitation  # scale操作

        return scale

    def xtresnet_se_block(self, inputs, channels, name):
        """
            功能: se_xtResnet模块
            输入:
                inputs: 输入特征
                name: tf变量名
            返回:
                h_out_2: 做XtResnet+SE操作后的特征
        """
        orig = tf.identity(inputs)  # 恒等映射

        # xtResnet模块的split层
        layers_split = list()
        for i in range(self.cardinality):
            w_conv_1 = initialize_weight(name + "_xt_conv_1_"+str(i), [3, 3, channels, int(channels/self.cardinality)])
            w_conv_1_key = self.get_bn_num()

            w_conv_2 = initialize_weight(name + "_xt_conv_2_"+str(i), [3, 3, int(channels/self.cardinality), channels])
            w_conv_2_key = self.get_bn_num()

            with tf.variable_scope(w_conv_1_key):
                conv_output1 = tf.nn.conv2d(inputs, w_conv_1, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')
                output_bn1 = tf.layers.batch_normalization(conv_output1, epsilon=1e-5, axis=1, fused=True,
                                                           center=True, scale=False, training=self.training,
                                                           reuse=self.variable_reused)
                output_1 = tf.nn.relu(output_bn1)

            with tf.variable_scope(w_conv_2_key):
                conv_output2 = tf.nn.conv2d(output_1, w_conv_2, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')
                output_bn2 = tf.layers.batch_normalization(conv_output2, epsilon=1e-5, axis=1, fused=True,
                                                           center=True, scale=False, training=self.training,
                                                           reuse=self.variable_reused)
                output_2 = tf.nn.relu(output_bn2)

            layers_split.append(output_2)

            self.add_conv_bn_weight(w_conv_1, w_conv_1_key)
            self.add_conv_bn_weight(w_conv_2, w_conv_2_key)

        xtresnet_out = tf.add_n(layers_split)

        # se模块
        h_out_se = self.squeeze_excitation_layer(xtresnet_out, self.FILTER_CHANNELS, self.reduction_ratio, name+'_SE_')
        h_out_2 = tf.nn.relu(tf.add(h_out_se, orig))

        return h_out_2

    def initialize_fc_and_bias_vars(self, fc_name, fc_shape, bias_name, bias_shape):
        """
            功能: 建立全连接网络
            输入:
                fc_name: 全连接层权重名
                fc_shape: 全连接层维度
                bias_name: 偏置层权重名
                bias_shape: 偏置层维度
            返回:
                h_fc1: 全连接层变量
                h_fc3: 偏置层变量
        """
        W_fc = initialize_weight(fc_name, fc_shape)
        b_fc = initialize_bias(bias_name, bias_shape)
        self.add_weights(W_fc)
        self.add_weights(b_fc)

        return W_fc, b_fc

    def construct_cnn_policy_value_net(self, planes):
        """
            功能: 建立深度学习网络，包括卷积模块，策略网络，和估值网络
            输入:
                planes: 棋谱数据
            返回:
                h_fc1: 策略网络得分
                h_fc3: 估值网络得分
        """
        print("construct_cnn_policy_value_net")
        x_planes = tf.reshape(planes, [-1, 18, 19, 19])  # 棋谱数据

        # 卷积网络部分
        flow = self.conv_and_bn_block(x_planes,
                                      filter_size=3, input_channels=18, output_channels=self.FILTER_CHANNELS,
                                      name="input_conv")
        flow = tf.nn.relu(flow)

        # xt_resnet_se部分
        for i in range(0, self.XT_SE_BLOCK_NUM):
            print("  xtResnet_se_"+str(i))
            block_name = "xtResnet_se_" + str(i)
            flow = self.xtresnet_se_block(flow, self.FILTER_CHANNELS, name=block_name)

        # 策略网络
        conv_pol = self.conv_and_bn_block(flow,
                                          filter_size=1, input_channels=self.FILTER_CHANNELS, output_channels=2,
                                          name="policy_head")
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 2 * 19 * 19])
        W_fc1, b_fc1 = self.initialize_fc_and_bias_vars("w_fc_1", [2 * 19 * 19, (19 * 19) + 1],
                                                        "b_fc_1", [(19 * 19) + 1])
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1)

        # 估值网络
        conv_val = self.conv_and_bn_block(flow,
                                          filter_size=1, input_channels=self.FILTER_CHANNELS, output_channels=1,
                                          name="value_head")
        h_conv_val_flat = tf.reshape(conv_val, [-1, 19 * 19])
        W_fc2, b_fc2 = self.initialize_fc_and_bias_vars("w_fc_2", [19 * 19, 256],
                                                        "b_fc_2", [256])
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3, b_fc3 = self.initialize_fc_and_bias_vars("w_fc_3", [256, 1],
                                                        "b_fc_3", [1])
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))

        print("finish construct_cnn_policy_value_net")
        return h_fc1, h_fc3

    def train_and_validate_process(self, batch_size, train_accuarcy_file):
        """
            功能: 网络训练和验证过程
            输入:
                batch_size: 批数据规模
            返回:
                无
        """
        if not self.time_start:
            self.time_start = time.time()
        policy_loss, mse_loss, reg_term, _, _ = self.session.run(
            [self.policy_loss, self.mse_loss, self.reg_term, self.train_op, self.next_batch],
            feed_dict={self.training: True, self.handle: self.train_handle})

        steps = tf.train.global_step(self.session, self.global_step)
        if steps % 200 == 0:
            print("  train step = " + str(steps))
        mse_loss = mse_loss / 4.0
        self.avg_policy_loss.append(policy_loss)
        self.avg_mse_loss.append(mse_loss)
        self.avg_reg_term.append(reg_term)
        if steps % self.train_swa_step == 0:
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                speed = batch_size * (1000.0 / elapsed)
            avg_policy_loss = np.mean(self.avg_policy_loss or [0])
            avg_mse_loss = np.mean(self.avg_mse_loss or [0])
            avg_reg_term = np.mean(self.avg_reg_term or [0])
            print("step {}, policy={:g} mse={:g} reg={:g} total={:g} ({:g} pos/s)".
                  format(steps, avg_policy_loss, avg_mse_loss, avg_reg_term,
                         avg_policy_loss + 1.0 * 4.0 * avg_mse_loss + avg_reg_term, speed))
            train_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Policy Loss", simple_value=avg_policy_loss),
                tf.Summary.Value(tag="MSE Loss", simple_value=avg_mse_loss)])
            self.train_log.add_summary(train_summaries, steps)
            self.time_start = time_end
            self.avg_policy_loss, self.avg_mse_loss, self.avg_reg_term = [], [], []
        if steps % (self.train_swa_step) == 0:  # 每隔一定步数就测试一次
            sum_accuracy = 0
            sum_mse = 0
            sum_policy = 0
            test_batches = 10
            for _ in range(0, test_batches):
                test_policy, test_accuracy, test_mse, _ = self.session.run(
                    [self.policy_loss, self.accuracy, self.mse_loss, self.next_batch],
                    feed_dict={self.training: False, self.handle: self.test_handle})
                sum_accuracy += test_accuracy
                sum_mse += test_mse
                sum_policy += test_policy

            sum_accuracy /= test_batches
            sum_policy /= test_batches
            sum_mse /= (4.0 * test_batches)
            test_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy", simple_value=sum_accuracy),
                tf.Summary.Value(tag="Policy Loss", simple_value=sum_policy),
                tf.Summary.Value(tag="MSE Loss", simple_value=sum_mse)])
            self.test_log.add_summary(test_summaries, steps)
            print("step {}, policy={:g} training accuracy={:g}%, mse={:g}".
                  format(steps, sum_policy, sum_accuracy * 100.0, sum_mse))
            train_acc_str = str(steps) + " " + str(sum_accuracy * 100.0) + "\n"
            train_accuarcy_file.write(train_acc_str)  # 将训练准确度保存到记事本中
            train_accuarcy_file.flush()
            weight_path = get_path("weights_file/")
            path = os.path.join(weight_path, "xtResSE-weights")
            weight_path = path + "-" + str(steps) + ".txt"  # 将tf权重保存成文本文件
            print("save_trained_weights")
            t = time.time()
            self.save_trained_weights(weight_path)
            showtime(t, "save_trained_weights")
            print("The weights are saved to {}".format(weight_path))

            if self.swa_enabled:  # 在_init_中给定
                t = time.time()
                self.save_smooth_weights(steps, path)  # 对权重做加权平均后保存
                showtime(t, "save_smoothed_trained_weights")

            save_path = self.saver.save(self.session, path, global_step=steps)
            print("Model saved in file: {}".format(save_path))

        if steps >= self.max_step:
            print("Reach Max Step {}. Training Complete.".format(self.max_step))
            train_accuarcy_file.write("\n")
            train_accuarcy_file.close()
            self.session.close()
            os._exit(0)
            print("OS EXIT!")

    def save_trained_weights(self, filename):
        """
        功能: 保存被训练的权重
        输入:
            filename: 文件名
        返回:
            无
        """
        with open(filename, "w") as file:
            file.write("1")  # 版本标示
            print("weights number:{:d}".format(len(self.weights)))
            for weights in self.weights:
                file.write("\n")  # 一个权重占一行
                work_weights = None
                if weights.name.endswith('/batch_normalization/beta:0'):
                    # 为向后兼容，在BN前将BN的beta值加上偏置，
                    var_key = weights.name.replace('beta', 'moving_variance')  # 替换变量键值
                    var = tf.get_default_graph().get_tensor_by_name(var_key)  # 找到该变量对应的张量
                    work_weights = tf.multiply(weights, tf.sqrt(var + tf.constant(1e-5)))
                elif weights.shape.ndims == 4:  # 对卷积权重的形状做转置为TF格式
                    # TF格式 [filter_height, filter_width, in_channels, out_channels]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:  # 对全连接权重的形状做转置
                    # TF格式 [in, out]
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    work_weights = weights  # 偏置, BN等
                nparray = work_weights.eval(session=self.session)
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                file.write(" ".join(wt_str))

    def save_smooth_weights(self, steps, path):
        """
        功能: 保存加权平均后的权重
        输入:
            steps: 训练步数
            swa_path: 加权平均后的权重文件路径
        返回:
            无
        """
        rem = self.session.run(tf.assign_add(self.swa_skip, -1))  # swa_skip在intialize_network函数中初始化为1
        if rem > 0:
            return
        self.swa_skip.load(self.swa_c, self.session)  # swa_skip在初始化时为1

        num = self.session.run(self.swa_accum_op)  # 运行加权累加操作的session

        if self.swa_max_n is not None:  # 初始化时为16
            num = min(num, self.swa_max_n)
            self.swa_count.load(float(num), self.session)

        swa_path = path + "-smoothed-" + str(int(num)) + "-" + str(steps) + ".txt"

        # 保存当前计算图中所有变量
        if not hasattr(self, 'save_op'):
            save_ops, rest_ops = [], []
            # rest_ops = []
            for var in self.weights:
                if isinstance(var, str):  # isinstance判断一个对象是否是一个已知的类型，类似type
                    var = tf.get_default_graph().get_tensor_by_name(var)
                name = var.name.split(':')[0]
                v = tf.Variable(var, name='save/' + name, trainable=False)  # 变量都为不可训练的
                save_ops.append(tf.assign(v, var))
                rest_ops.append(tf.assign(var, v))
            self.save_op = tf.group(*save_ops)
            self.restore_op = tf.group(*rest_ops)
        self.session.run(self.save_op)

        self.session.run(self.swa_load_op)  # 将加权平均后的权重变量拷贝到当前图中
        if self.swa_recalc_bn:  # 在_init()_中初始化
            print("Smoothing the values of BN")
            for _ in range(200):
                self.session.run([self.loss, self.update_ops, self.next_batch],
                                 feed_dict={self.training: True, self.handle: self.train_handle})

        self.save_trained_weights(swa_path)
        self.session.run(self.restore_op)  # 恢复当前计算图中所有变量

        print("The smoothed weights are saved in {}".format(swa_path))

    def restore_tf_graph(self, file_name):
        """
        功能: 从检查点开始继续训练
        输入:
            file_name: 权重文件名
        返回:
            无
        """
        print("Restoring from {0}".format(file_name))

        reader = tf.train.NewCheckpointReader(file_name)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0])
                            for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_default_graph().get_tensor_by_name(var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
        opt_saver = tf.train.Saver(restore_vars)
        opt_saver.restore(self.session, file_name)
