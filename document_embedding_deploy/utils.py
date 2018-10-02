import tensorflow as tf
import glob
import collections
from itertools import chain
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def build_dataset(sentences, vocabulary_size):
    words = []
    for sent in sentences:
        for splited_sent in sent.split(' '):
            words.append(splited_sent)
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    unk_count = 0
    sent_data = []
    for sentence in sentences:
        data = []
        for word  in sentence.split():
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count = unk_count + 1
            data.append(index)
        sent_data.append(data)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # sent_data = 단어(key)에 대한 value값으로 구성된 하나의 리스트
    # count = 각 단어들이 총 문서에 몇 번씩 등장하였는지
    # dictionary = {단어: value}
    # reverse_dictionary = {value: 단어}
    return sent_data, count, dictionary, reverse_dictionary