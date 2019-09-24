from predict import Predictor

from flask import Flask, jsonify
from flask import request
import json
from gevent.pywsgi import WSGIServer
import time

app = Flask(__name__)

src_vocab_path = "checkpoint/vocab.src"
tgt_vocab_path = "checkpoint/vocab.tgt"
model_path = "checkpoint/aivivn_tone.model.ep25"
wlm_path = "lm/corpus-wplm-4g-v2.binary"

predictor = Predictor(src_vocab_path, tgt_vocab_path, model_path, wlm_path)

def tone(predictor, data_type):
    if request.method == "POST":
        dataDict = json.loads(request.data.decode('utf-8'))
        input = dataDict.get(data_type, None)

    start = time.time()

    result = predictor.infer(input)

    end = time.time() - start
    response = jsonify({"result": result, "time": end, "status_code": 200})
    response.status_code = 200
    response.status = 'OK'
    return response, 200


@app.route('/tone', methods=['POST'])
def tone_():
    return tone(predictor, data_type="text")

if __name__ == "__main__":
    http_server = WSGIServer(('', 4000), app)
    http_server.serve_forever()

