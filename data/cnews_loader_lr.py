# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np
import os
import re

base_dir = 'data'
stopwords_dir = os.path.join(base_dir, 'stop_words.txt')
train_dir = os.path.join(base_dir + '/cnews', 'cnews.train.txt')
test_dir = os.path.join(base_dir + '/cnews', 'cnews.test.txt')
val_dir = os.path.join(base_dir + '/cnews', 'cnews.val.txt')
ori_dir = os.path.join(base_dir + '/cnews', 'cnews.corpus.txt')


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def stopwords_list(filename):
    stopwords = []
    with open_file(filename) as f:
        for line in f:
            try:
                content = line.strip()
                stopwords.append(content)
            except:
                pass
    return stopwords


stopwords = stopwords_list(stopwords_dir)


def remove_stopwords(content):
    return list(set(content).difference(set(stopwords)))


# model = gensim.models.Word2Vec.load("news_12g_baidubaike_20g_novel_90g_embedding_64.model")


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                contents.append(content)
                labels.append(label)
            except:
                pass
    return contents, labels


def number_norm(x):
    if re.compile(r'^\d*$').match(x):
        if len(x) >= 16:
            return '0000000000000000'
        elif 8 <= len(x) < 16:
            return '00000000'
        else:
            return '00'
    return x


def build_vocab(total_dir, vocab_dir):
    """根据训练集构建词汇表，存储"""
    print("building vacab...")
    words = []
    with open_file(total_dir) as f:
        for line in f:
            sents = list(line.strip().split('\t')[1])
            for sent in sents:
                # words.extend(number_norm(sent))
                words.extend(sent)
    words = remove_stopwords(words)
    # counter = Counter(words)
    # count_pairs = counter.most_common(5000)
    # words, _ = list(zip(*count_pairs))
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def seg_length(seg_dir):
    """得到分词长度"""
    seg_length = 0
    with open_file(seg_dir) as f:
        for line in f:
            try:
                if not line.strip():
                    seg_length += 1
            except:
                pass
    return seg_length


def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(0, len(words))))

    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    # categories = ['体育', '财经', '房产', '家居',
    #     '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_set = set()
    with open_file(val_dir) as f:
        for line in f:
            cat_set.add(line.split("\t")[0].strip())
    categories = list(cat_set)
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


# def process_sent(words, word_to_id, max_sent_length):
#     data_id = []
#     for i in range(len(words)):
#         data_id.append([word_to_id[x] for x in words if x in word_to_id])
#     # 使用keras提供的pad_sequences来将文本pad为固定长度
#     x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max(max_sent_length, 8))
#     return x_pad


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
    """Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    Arguments:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)

    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    Arguments:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    Returns:
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def process_file(filename, word_to_id, cat_to_id, vocab_size):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data = np.zeros((len(contents), vocab_size)).tolist()
    label = []
    for i in range(len(contents)):
        words = list(contents[i].strip())
        words = remove_stopwords(words)
        dd = [word_to_id[x] for x in words if x in word_to_id]
        counter = Counter(dd)
        for k, v in counter.items():
            data[i][k] = v
        label.append(cat_to_id[labels[i]])
    return data, to_categorical(label)


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = []
    y_shuffle = []
    for i in range(len(indices)):
        x_shuffle.append(x[indices[i]])
        y_shuffle.append(y[indices[i]])

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
