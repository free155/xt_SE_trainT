"""
    功能:数据处理程序
    generate_train_and_test_chunk (chunks, test_ratio): 将数据集划分成训练集合测试集两部分
    map_data_chunk_to_tensor (planes, probs, winner): 解析棋谱数据
    map_origcord_into_newcoord(vertex, symmetry): 将落子的坐标根据对称性重新映射为新的落子坐标
    generate_tf_data_iterator(data_chunk): 生成tf训练用的数据迭代器

    OriginalTrainData: 原始棋谱训练数据源
    ChunkDataSrc类: 从棋谱数据源中生成数据块
    TempBufShufl类: 开辟临时缓存，并将缓存中的数据打乱
    ChunkParser类: 解析棋谱chunk
"""

import binascii
import gzip
import itertools
import math
import queue
import struct
import threading
import unittest

from common import *


def generate_train_and_test_chunk(chunks, test_ratio):
    """
        功能: 将数据集划分成训练集合测试集两部分
        输入:
            chunks: 整个数据集
            test_ratio: 测试集所占比例
        返回:
            chunks[:splitpoint]: 训练集
            chunks[splitpoint:]: 测试集
    """
    splitpoint = 1 + int(len(chunks) * (1.0 - test_ratio))

    return (chunks[:splitpoint], chunks[splitpoint:])


def map_data_chunk_to_tensor(planes, probs, winner):
    """
        功能: 解析棋谱数据
        输入：
            planes: 棋谱
            probs:  走子概率
            winner：胜率
        返回：
            planes: 棋谱tensor
            probs:  走子概率tensor
            winner：胜率tensor
    """
    planes = tf.decode_raw(planes, tf.uint8)
    probs = tf.decode_raw(probs, tf.float32)
    winner = tf.decode_raw(winner, tf.float32)

    planes = tf.to_float(planes)

    planes = tf.reshape(planes, (BATCH_NUM, 18, 19 * 19))
    probs = tf.reshape(probs, (BATCH_NUM, 19 * 19 + 1))
    winner = tf.reshape(winner, (BATCH_NUM, 1))

    return (planes, probs, winner)


def map_origcord_into_newcoord(vertex, symmetry):
    """
        功能: 将落子的坐标根据对称性重新映射为新的落子坐标
        输入：
            vertex: 顶点数据
            symmetry:  对称值
        返回：
            重映射后的坐标
    """
    assert vertex >= 0 and vertex < 361
    x = vertex % 19
    y = vertex // 19
    if symmetry >= 4:
        x, y = y, x
        symmetry -= 4
    if symmetry == 1 or symmetry == 3:
        x = 19 - x - 1
    if symmetry == 2 or symmetry == 3:
        y = 19 - y - 1
    return y * 19 + x


def generate_tf_data_iterator(data_chunk):
    """
        功能: 生成tf训练用的数据迭代器
        输入:
            data_chunk: 数据块
        返回:
            dataset: 数据集
            tf_iterator: 迭代器
    """
    data_parser = GenDataBatch(OriginalTrainData(data_chunk), shuffle_size=1 << 19,
                               sample=DOWN_SAMPLE_RATIO,
                               batch_size=BATCH_NUM)
    dataset = tf.data.Dataset.from_generator(data_parser.parse_data_chunk,
                                             output_types=(tf.string, tf.string, tf.string))  # TF1.4 新特性
    dataset = dataset.map(map_data_chunk_to_tensor)
    dataset = dataset.prefetch(4)
    tf_iterator = dataset.make_one_shot_iterator()

    return dataset, tf_iterator


class OriginalTrainData:
    """
        棋谱数据源解析.
    """
    def __init__(self, chunks):
        self.chunks = []
        self.done = chunks

    def next(self):
        if not self.chunks:
            self.chunks, self.done = self.done, self.chunks
            random.shuffle(self.chunks)
        if not self.chunks:
            return None
        while len(self.chunks):
            filename = self.chunks.pop()
            try:
                with gzip.open(filename, 'rb') as chunk_file:
                    self.done.append(filename)
                    return chunk_file.read()
            except:
                print("failed to parse {}".format(filename))


class TempBufShufl:
    """
        开辟缓存数据，并将缓存中的数据打乱.
    """
    def __init__(self, elem_size, elem_count):
        assert elem_size > 0, elem_size
        assert elem_count > 0, elem_count
        self.elem_size = elem_size
        self.elem_count = elem_count
        self.buffer = bytearray(elem_size * elem_count)
        self.used = 0

    def get_buffer_data(self):
        if self.used < 1:
            return None
        self.used -= 1
        i = self.used
        return self.buffer[i * self.elem_size: (i+1) * self.elem_size]

    def shuffle_operation(self, item):
        assert len(item) == self.elem_size, len(item)

        if self.used > 0:
            i = random.randint(0, self.used-1)
            old_item = self.buffer[i * self.elem_size: (i+1) * self.elem_size]
            self.buffer[i * self.elem_size: (i+1) * self.elem_size] = item
            item = old_item
        if self.used < self.elem_count:
            i = self.used
            self.buffer[i * self.elem_size: (i+1) * self.elem_size] = item
            self.used += 1
            return None
        return item


class GenDataBatch:
    """
        解析棋谱chunk.
    """
    def __init__(self, chunkdatasrc, shuffle_size=1, sample=1, batch_size=256, workers=None):
        self.prob_reflection_table = [[map_origcord_into_newcoord(vertex, sym) for vertex in range(361)]+[361] for sym in range(8)]
        self.full_reflection_table = [
            np.array([map_origcord_into_newcoord(vertex, sym) + p * 361 for p in range(16) for vertex in range(361)]) for sym in range(8)]
        self.prob_reflection_table = [np.array(x, dtype=np.int64) for x in self.prob_reflection_table]
        self.full_reflection_table = [np.array(x, dtype=np.int64) for x in self.full_reflection_table]
        self.flat_planes = [b'\1'*361 + b'\0'*361, b'\0'*361 + b'\1'*361]

        self.sample = sample
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        if workers is None:
            workers = max(1, mp.cpu_count() - 12)  # mp.cpu_count()得到的是逻辑核的数量
        print("Forking {} slave threads.".format(workers))

        self.readers = []
        for _ in range(workers):
            read, write = mp.Pipe(duplex=False)
            mp.Process(target=self.perform_task, args=(chunkdatasrc, write)).start()
            self.readers.append(read)
            write.close()
        self.init_structs()

    def init_structs(self):
        self.v2_struct = struct.Struct('4s1448s722sBB')
        self.raw_struct = struct.Struct('4s1448s6498s')

    def convert_itemdata(self, text_item):
        planes = []
        for plane in range(0, 16):
            hex_string = text_item[plane][0:90]
            array = np.unpackbits(np.frombuffer(bytearray.fromhex(hex_string), dtype=np.uint8))
            last_digit = text_item[plane][90]
            assert last_digit == "0" or last_digit == "1"
            planes.append(array)
            planes.append(np.array([last_digit], dtype=np.uint8))

        planes = np.concatenate(planes)
        planes = np.packbits(planes).tobytes()

        stm = text_item[16][0]
        assert stm == "0" or stm == "1"
        stm = int(stm)

        probabilities = np.array(text_item[17].split()).astype(np.float32)
        if np.any(np.isnan(probabilities)):
            return False, None
        assert len(probabilities) == 362

        probs = probabilities.tobytes()
        assert(len(probs) == 362 * 4)

        winner = float(text_item[18])
        assert winner == 1.0 or winner == -1.0
        winner = int((winner + 1) / 2)

        version = struct.pack('i', 1)

        return True, self.v2_struct.pack(version, probs, planes, stm, winner)

    def parse_item(self, symmetry, content):
        assert symmetry >= 0 and symmetry < 8

        (ver, probs, planes, to_move, winner) = self.v2_struct.unpack(content)

        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8))
        planes = planes[self.full_reflection_table[symmetry]]
        assert len(planes) == 19*19*16
        planes = np.packbits(planes)
        planes = planes.tobytes()

        probs = np.frombuffer(probs, dtype=np.float32)
        probs = probs[self.prob_reflection_table[symmetry]]
        assert len(probs) == 362
        probs = probs.tobytes()

        return self.v2_struct.pack(ver, probs, planes, to_move, winner)

    def convert_to_gen(self, content):
        (ver, probs, planes, to_move, winner) = self.v2_struct.unpack(content)
        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8))
        assert len(planes) == 19*19*16
        stm = to_move
        assert stm == 0 or stm == 1
        planes = planes.tobytes() + self.flat_planes[stm]
        assert len(planes) == (18 * 19 * 19), len(planes)

        winner = float(winner * 2 - 1)
        assert winner == 1.0 or winner == -1.0, winner
        winner = struct.pack('f', winner)

        return (planes, probs, winner)

    def convert_item(self, chunkdata):
        if chunkdata[0:4] == b'\1\0\0\0':
            print("V2 chunkdata")
            for i in range(0, len(chunkdata), self.v2_struct.size):
                if self.sample > 1:
                    if random.randint(0, self.sample-1) != 0:
                        continue
                yield chunkdata[i:i+self.v2_struct.size]
        else:
            print("V1 chunkdata")
            file_chunkdata = chunkdata.splitlines()
            for i in range(0, len(file_chunkdata), DATA_ITEM_LINES):
                if self.sample > 1:
                    if random.randint(0, self.sample-1) != 0:
                        continue
                item = file_chunkdata[i:i+DATA_ITEM_LINES]
                str_items = [str(line, 'ascii') for line in item]
                success, data = self.convert_itemdata(str_items)
                if success:
                    yield data

    def perform_task(self, chunkdatasrc, writer):
        self.init_structs()
        while True:
            chunkdata = chunkdatasrc.next()
            if chunkdata is None:
                break
            for item in self.convert_item(chunkdata):
                symmetry = random.randrange(8)
                item = self.parse_item(symmetry, item)
                writer.send_bytes(item)

    def generate_chunk_from_buffer(self):
        sbuff = TempBufShufl(self.v2_struct.size, self.shuffle_size)
        while len(self.readers):
            for r in self.readers:
                try:
                    s = r.recv_bytes()
                    s = sbuff.shuffle_operation(s)
                    if s is None:
                        continue
                    yield s
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
        while True:
            s = sbuff.get_buffer_data()
            if s is None:
                return
            yield s

    def generate_tuple_from_chunk(self, gen):
        for r in gen:
            yield self.convert_to_gen(r)

    def generate_batch_from_tuple(self, gen):
        while True:
            s = list(itertools.islice(gen, self.batch_size))
            if not len(s):
                return
            yield (b''.join([x[0] for x in s]), b''.join([x[1] for x in s]), b''.join([x[2] for x in s]))

    def parse_data_chunk(self):
        gen = self.generate_chunk_from_buffer()  # 从工作线程中读数据
        gen = self.generate_tuple_from_chunk(gen)  # 将v2版本的数据转换成元祖
        gen = self.generate_batch_from_tuple(gen)  # 将元祖数据打包成batch
        for b in gen:
            yield b
