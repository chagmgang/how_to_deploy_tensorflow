from konlpy.tag import Twitter
import numpy as np

# '<start>', '<end>', '<pad>', '<%>'

twitter = Twitter()

vocab_list = []
with open('input.log', 'r', encoding='utf-8') as content_file:
    for line in content_file:
        tag = twitter.pos(line)[:-1]
        for t in tag:
            vocab_list.append(t[0])
with open('output.log', 'r', encoding='utf-8') as content_file:
    for line in content_file:
        tag = twitter.pos(line)[:-1]
        for t in tag:
            vocab_list.append(t[0])

vocab_list = list(set(vocab_list))
vocab_list.append('<start>')
vocab_list.append('<end>')
vocab_list.append('<pad>')
vocab_list.append('<%>')

with open('vocab.log', 'w') as vocab_file:
    for w in vocab_list:
        vocab_file.write(w + '\n')