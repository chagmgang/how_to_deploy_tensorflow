from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

x, y = mnist.train.next_batch(10)

df = pd.DataFrame(x)
df.to_csv('x.csv', header=None, index=None)

df = pd.DataFrame(y)
df.to_csv('y.csv', header=None, index=None)

#x = pd.read_csv('x.csv', header=None)
#x = x.as_matrix()