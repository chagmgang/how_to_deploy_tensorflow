import tensorflow as tf
import numpy as np
from attention_model import s2s
from konlpy.tag import Twitter

twitter = Twitter()

input_sent = []
with open('input.log', 'r', encoding='utf-8') as content_file:
    for line in content_file:
        tag = twitter.pos(line)[:-1]
        input_sent.append([i[0] for i in tag])

output_sent = []
with open('output.log', 'r', encoding='utf-8') as content_file:
    for line in content_file:
        tag = twitter.pos(line)[:-1]
        output_sent.append([i[0] for i in tag])

vocab_list = []
with open('vocab.log', 'r', encoding='utf-8') as content_file:
    for line in content_file:
        vocab_list.append(line[:-1])

vocab_dict = {n: i for i, n in enumerate(vocab_list)}
num_dic = len(vocab_dict)


input_length = [len(i) for i in input_sent]
output_length = [len(o) for o in output_sent]
max_len_i = max(input_length)
max_len_o = max(output_length)

input_batch = []
output_batch = []
target_batch = []

for i, o in zip(input_sent, output_sent):
    while not len(i) == max_len_i:
        if len(i) < max_len_i:
            i.append('<%>')
    while not len(o) == max_len_o:
        if len(o) < max_len_o:
            o.append('<%>')
    real_i = i
    real_o = ['<start>'] + [x for x in o]
    real_target = [x for x in o] + ['<end>']

    input = [vocab_dict[n] for n in real_i]
    output = [vocab_dict[n] for n in real_o]
    target = [vocab_dict[n] for n in real_target]

    input_batch.append(np.eye(num_dic)[input])
    output_batch.append(np.eye(num_dic)[output])
    target_batch.append(target)


total_epoch = 300
n_class = n_input = num_dic

enc_sent_size = max_len_i + 1
output_sent_size = max_len_o + 1
vocab_size = num_dic

S2S = s2s(enc_sent_size, output_sent_size, vocab_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


for epoch in range(total_epoch):
    _, loss = sess.run([S2S.optimizer, S2S.cost],
                       feed_dict={S2S.enc_input: input_batch,
                                  S2S.dec_input: output_batch,
                                  S2S.targets: target_batch})

    print('Epoch:', '%05d' % (epoch + 1),
          'cost =', '{:.10f}'.format(loss))

print('최적화 완료!')



output_sent = []
for i in input_batch:
    output = ['<pad>' for i in range(output_sent_size - 1)]
    output_sent.append(output)


input_result = []
output_result = []
target_result = []

for i, o in zip(input_sent, output_sent):
    while not len(i) == max_len_i:
        if len(i) < max_len_i:
            i.append('<%>')
    while not len(o) == max_len_o:
        if len(o) < max_len_o:
            o.append('<%>')
    real_i = i
    real_o = ['<start>'] + [x for x in o]
    real_target = [x for x in o] + ['<end>']

    input = [vocab_dict[n] for n in real_i]
    output = [vocab_dict[n] for n in real_o]
    target = [vocab_dict[n] for n in real_target]

    input_result.append(np.eye(num_dic)[input])
    output_result.append(np.eye(num_dic)[output])
    target_result.append(target)

print(input_result)
print(output_result)
print(target_result)

result = sess.run(S2S.prediction,
                      feed_dict={S2S.enc_input: input_result,
                                 S2S.dec_input: output_result,
                                 S2S.targets: target_result})

for inp in input_result:
    x = []
    for i in inp:
        for name, age in vocab_dict.items():
            y = np.argmax(i)
            if y == age:
                x.append(name)
    x = ' '.join(x)
    x = x.replace('<%>', '')
    x = x.replace('<end>', '')
    print('질문:', x)

print('--------------')

for target in target_batch:
    x = []
    for tar in target:
        for name, age in vocab_dict.items():
            if tar == age:
                x.append(name)
    x = ' '.join(x)
    x = x.replace('<%>', '')
    x = x.replace('<end>', '')
    print('답변:', x)

print('--------------')

for re in result:
    x = []
    for r in re:
        for name, age in vocab_dict.items():
            if r == age:
                x.append(name)
    x = ' '.join(x)
    x = x.replace('<%>', '')
    x = x.replace('<end>', '')
    print('답변:', x)

SAVE_PATH = './save'
MODEL_NAME = 'test'
path = saver.save(sess, SAVE_PATH + '/' + MODEL_NAME + '.ckpt')
print("saved at {}".format(path))