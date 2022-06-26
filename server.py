from flask import Flask, request, make_response
from flask_restx import Resource, Api, reqparse
from flask_cors import CORS
import json
import boto3
from datetime import datetime
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.silence import split_on_silence
# import io
# import collections
# import contextlib
# import sys
# import wave
# import webrtcvad
# import os
# import ssl
import werkzeug
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime
# import datetime
import json
# import requests
from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.ai.anomalydetector.models import DetectRequest, TimeSeriesPoint, TimeGranularity, \
    AnomalyDetectorError
from azure.core.credentials import AzureKeyCredential
import pandas as pd
from dateutil import parser
import time
import anomaly


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

CORS(app, resources={r'/*': {'origins': '*'}})

SUBSCRIPTION_KEY = '20fb2829f07c4018938ce3a45fc0bcd1'
ANOMALY_DETECTOR_ENDPOINT = 'https://v-dat.cognitiveservices.azure.com/anomalydetector/v1.1-preview/timeseries/entire/detect'


@api.route('/hello')  # 데코레이터 이용, '/hello' 경로에 클래스 등록
class HelloWorld(Resource):
    def get(self):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        print(request.json)
        return {"This is": "Test!"}


@api.route('/upload')
class Upload(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument(
            'file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parser.parse_args()
        print(parser)
        print(args)
        file_object = args['file']


@api.route('/sensor')
class GetAnomaly(Resource):
    def get(self):
        a = anomaly.VDAT()
        sensor = "../20220611/지용님/20220611_142409_data(head,e4,eye).csv"
        volume = "../20220611/지용님/20220611_143903_voice.wav"
        result = a.getSensorResult(sensor, volume)

        return result


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port="8080"
    )
