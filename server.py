from flask import Flask, request, jsonify
from flask_restx import Resource, Api
from flask_cors import CORS
from io import BytesIO
# import collections
# import contextlib
# import sys
# import wave
# import webrtcvad
# import os
# import ssl
import anomaly
import pandas as pd


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

CORS(app, resources={r'/*': {'origins': '*'}})

SUBSCRIPTION_KEY = '20fb2829f07c4018938ce3a45fc0bcd1'
ANOMALY_DETECTOR_ENDPOINT = 'https://v-dat.cognitiveservices.azure.com/anomalydetector/v1.1-preview/timeseries/entire/detect'


@api.route('/hello')  # 데코레이터 이용, '/hello' 경로에 클래스 등록
class HelloWorld(Resource):
    def get(self):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return {"This is": "Test!"}


@api.route('/upload')
class UploadTest(Resource):
    def post(self):
        f = request.files['file'].read()
        voice = request.form['voice']
        df = pd.read_csv(BytesIO(f))


@api.route('/sensor')
class GetAnomaly(Resource):
    def post(self):
        a = anomaly.VDAT()
        f = request.files['file'].read()
        volume = request.form['voice']
        result = a.getSensorResult(f, volume)

        return result


if __name__ == '__main__':
    app.run(
        debug=True,
        host="0.0.0.0",
        port="8081"
    )
