import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from predict_client.prod_client import ProdClient

HOST = '0.0.0.0:9000'
# a good idea is to place this global variables in a shared file
MODEL_NAME = 'model'
MODEL_VERSION = 1

client = ProdClient(HOST, MODEL_NAME, MODEL_VERSION)


img = plt.imread('mini/trainA/n02381460_1524.jpg')
img = img / 255
img = np.array([img])
print(img)

req_data = [{'in_tensor_name': 'inputs', 'in_tensor_dtype': 'DT_FLOAT', 'data': img}]

prediction = client.predict(req_data, request_timeout=10)

print(prediction)