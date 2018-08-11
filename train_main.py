"""
    功能: 棋谱训练主程序
    main (): 主函数
"""

from common import *
import dataprocess as dp
from xt_se_process import XTSETensorFlow


def main(args):
    """
        功能: 主函数
        args: 命令行输入，设置训练参数、给定训练数据源
    """
    initial_lr = float(args.pop(0))  # 初始学习率
    max_step = int(args.pop(0))  # 最大训练步数
    train_swa = int(args.pop(0))  # 是否保存权重的加权平均
    train_swa_step = int(args.pop(0))  # 对权重做加权平均的步数
    train_data_dir = args.pop(0)

    data_chunks = glob.glob(train_data_dir + "*.gz")  # 读取训练集

    print("The train data set is in:")
    print(train_data_dir)
    if not data_chunks:
        print("no data chunk and exit")
        return
    else:
        print("There are {0} data chunks. ".format(len(data_chunks)))

    random.shuffle(data_chunks)  # 打乱数据集
    training_chunks, test_chunks = dp.generate_train_and_test_chunk(data_chunks, 0.1)
    print("The number of training chuns is {0}, and the number of test chunk is {1}. ".\
          format(len(training_chunks), len(test_chunks)))

    #  解析训练数据chunk，生成训练迭代器
    data_set, train_iterator = dp.generate_tf_data_iterator(training_chunks)

    #  解析测试数据chunk，生成测试迭代器
    data_set, test_iterator = dp.generate_tf_data_iterator(test_chunks)

    tfprocess = XTSETensorFlow(initial_lr, max_step, train_swa, train_swa_step)  # 训练参数初始化
    tfprocess.init_handle(data_set, train_iterator, test_iterator)  # tf计算图初始化

    if args:
        print("continue to train from check point")
        restore_file = args.pop(0)
        tfprocess.restore_tf_graph(restore_file)  # 从检查点开始继续训练
    print("begin train_and_validate_process")
    f = open("train_accuary_result.txt", "a+")
    while True:
        tfprocess.train_and_validate_process(BATCH_NUM, f)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main(sys.argv[1:])
    mp.freeze_support()
