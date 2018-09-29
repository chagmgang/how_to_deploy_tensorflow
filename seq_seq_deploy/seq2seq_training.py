# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.
# 영어 단어를 한국어 단어로 번역하는 프로그램을 만들어봅니다.
import tensorflow as tf
import numpy as np
from seq2seq_model import s2s

# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       word -> ['w', 'o', 'r', 'd']
#       to   -> ['t', 'o', 'P', 'P']
char_arr = [c for c in '%SEPwordgameil단어나무놀이소녀']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 영어를 한글로 번역하기 위한 학습 데이터
seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀']]


def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    len_i = [len(seq[0]) for seq in seq_data]
    len_o = [len(seq[1]) for seq in seq_data]

    max_len_i = max(len_i)
    max_len_o = max(len_o)

    for seq in seq_data:
        i = seq[0]
        o = seq[1]
        while not len(i) == max_len_i:
            if len(i) < max_len_i:
                i += '%'
        while not len(o) == max_len_o:
            if len(o) < max_len_o:
                o += '%'

    
        input = [num_dic[n] for n in i]
        output = [num_dic[n] for n in ('S' + o)]
        target = [num_dic[n] for n in (o + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)

    return input_batch, output_batch, target_batch, max_len_i + 1, max_len_o + 1, dic_len

learning_rate = 0.01
n_hidden = 128
total_epoch = 500
n_class = n_input = dic_len

input_batch, output_batch, target_batch, enc_sent_size, output_sent_size, vocab_size = make_batch(seq_data)
S2S = s2s(enc_sent_size, output_sent_size, vocab_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for epoch in range(total_epoch):
    _, loss = sess.run([S2S.optimizer, S2S.cost],
                       feed_dict={S2S.enc_input: input_batch,
                                  S2S.dec_input: output_batch,
                                  S2S.targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')

DATA_SIZE = 100
SAVE_PATH = './save'
EPOCHS = 400
LEARNING_RATE = 0.01
MODEL_NAME = 'test'
path = saver.save(sess, SAVE_PATH + '/' + MODEL_NAME + '.ckpt')
print("saved at {}".format(path))