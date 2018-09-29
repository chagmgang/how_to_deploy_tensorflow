import numpy as np
from predict_client.prod_client import ProdClient

HOST = '0.0.0.0:9000'
# a good idea is to place this global variables in a shared file
MODEL_NAME = 'test'
MODEL_VERSION = 1

client = ProdClient(HOST, MODEL_NAME, MODEL_VERSION)

x=np.array([[[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.]]])
y=np.array([[[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.]]])
z=np.array([[3, 3, 2]])

req_data = [{'in_tensor_name': 'inputs', 'in_tensor_dtype': 'DT_FLOAT', 'data': x},
            {'in_tensor_name': 'outputs', 'in_tensor_dtype': 'DT_FLOAT', 'data': y},
            {'in_tensor_name': 'targets', 'in_tensor_dtype': 'DT_FLOAT', 'data': z}]

prediction = client.predict(req_data, request_timeout=10)

print(prediction)