# how_to_deploy_tensorflow


## Make toy tensorflow model

```
python toy_model.py
```

## Fetch the variables you need from your stored model(.ckpt)

```
python serve.py
```

## Download a Docker image with TensorFlow serving already compile on it

```
docker run -it -p <PORT>:<PORT> — name <CONTAINER_NAME> -v <ABSOLUTE_PATH_TO_SERVE_DIR>:<SERVE_DIR> epigramai/model-server:light-universal-1.7 — port=<PORT> — model_name=<MODEL_NAME> — model_base_path=<MODEL_DIR>
```

for example


```
docker run -it -p 9000:9000 --name tf-serve -v $(pwd)/serve/:/serve/ epigramai/model-server:light-universal-1.7 --port=9000 --model_name=test --model_base_path=/serve/test
```

or

```
docker run -it -p 9000:9000 --name tf-serve -v $(pwd)/serve/:/serve/ chagmgang/tensorflow_deploy:1.7 --port=9000 --model_name=test --model_base_path=/serve/test
```

## Create a client in order to send the gRPC request

Install library python for make client and end-point

```
pip install git+https://github.com/chagmgang/tfserving-python-predict-client.git
```

or

```
pip install git+https://github.com/epigramai/tfserving-python-predict-client.git
```
## Test the response of docker

```
python client.py
```

## Make a API endpoint via Flask

```
import numpy as np
from predict_client.prod_client import ProdClient
from flask import Flask
from flask import request
from flask import jsonify

HOST = 'localhost:9000'
MODEL_NAME = 'test'
MODEL_VERSION = 1

app = Flask(__name__)
client = ProdClient(HOST, MODEL_NAME, MODEL_VERSION)

def convert_data(raw_data):
    return np.array(raw_data, dtype=np.float32)

def get_prediction_from_model(data):
    req_data = [{'in_tensor_name': 'inputs', 'in_tensor_dtype': 'DT_FLOAT', 'data': data}]

    prediction = client.predict(req_data, request_timeout=10)

    return prediction


@app.route("/prediction", methods=['POST'])
def get_prediction():
    req_data = request.get_json()
    raw_data = req_data['data']

    data = convert_data(raw_data)
    prediction = get_prediction_from_model(data)

    # ndarray cannot be converted to JSON
    return jsonify({ 'predictions': prediction['outputs'].tolist() })

if __name__ == '__main__':
    app.run(host='localhost',port=3000)
```
or
```
python api_endpoint.py
```

## test via postman
![image](./src/1_OxCFwpBMGspD_AwEuaZwxQ.png)
