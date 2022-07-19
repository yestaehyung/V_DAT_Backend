import pandas as pd
import json
import boto3
import datetime
import os
import numpy as np
import re
from urllib import request
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import io
# import collections
# import contextlib
# import sys
import wave
# import math
# import webrtcvad
# from scipy.io import wavfile
import ssl
import requests
from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.ai.anomalydetector.models import DetectRequest, TimeSeriesPoint, TimeGranularity, \
    AnomalyDetectorError
from azure.core.credentials import AzureKeyCredential
import pandas as pd
# from dateutil import parser


import time

SUBSCRIPTION_KEY = '20fb2829f07c4018938ce3a45fc0bcd1'
ANOMALY_DETECTOR_ENDPOINT = 'https://v-dat.cognitiveservices.azure.com/anomalydetector/v1.1-preview/timeseries/entire/detect'


class VDAT():

    def __init__(self):
        self.df = None
        self.vrTime = None
        self.eyeTrack = None
        self.pHmd = None
        self.pEyetracker = None
        self.pBvp = None
        self.pEda = None
        self.pTmp = None
        self.pIbi = None
        self.pHr = None
        self.requestPosX = None
        self.requestPosY = None
        self.requestPosZ = None
        self.requestCombinedX = None
        self.requestCombinedY = None
        self.requestCombinedZ = None
        self.requestBvp = None
        self.requestEda = None
        self.requestTmp = None
        self.requestBvp = None
        self.requestIbi = None
        self.requestHr = None

    def csv2df(self, csv):

        data = pd.read_csv(io.BytesIO(csv))
#         data = pd.read_csv(csv)
        data = data[data[' Start'] == 1].reset_index(drop=True)
        data = data.fillna(0)
#         data = data.reset_index(drop=True)
        data['time'] = [i.replace(" ", "") for i in data[' Time']]
        data['unix'] = [time.mktime(datetime.datetime.strptime(str(data['Date'][i])+data['time'][i], '%Y%m%d%H:%M:%S').timetuple())
                        for i in range(0, len(data))]
        datatime = data[['Date', 'time']]
        vrDate = datatime['Date'][0]
        vrTime = []
        for i in sorted(datatime['time']):
            t = datetime.datetime.strptime(
                str(vrDate) + ' ' + i, "%Y%m%d %H:%M:%S")
            vrTime.append(t.isoformat())
        data['vrTime'] = vrTime

        data = data.drop(' Time', axis=1)

        return data

    def check_volume(self, vol):
        threshold = [-35, -15]

        if vol <= threshold[0]:
            return -1
        elif vol <= threshold[1]:
            return 0
        else:
            return 1

    def getVolume(self, name):

        # path from URL
        url = "https://dt1amnyxy57si.cloudfront.net/audios/" + name
        Freq = 16000
        context = ssl._create_unverified_context()
        audio_bytes = request.urlopen(url, context=context).read()

        # path from local
        # audio_bytes = wave.open(link, 'rb')
        # Freq = 16000
        # audio_bytes = audio_bytes.readframes(14400000)

        # Convert wav to audio_segment
        audio_segment = AudioSegment.from_raw(
            io.BytesIO(audio_bytes),
            sample_width=2,
            frame_rate=Freq,
            channels=1
        )

        nonsilent_data = detect_nonsilent(
            audio_segment,
            min_silence_len=600,
            silence_thresh=-37,
            seek_step=1
        )

        remove_list = []
        for i in nonsilent_data:
            if ((i[1] - i[0]) < 350):
                remove_list.append(i)
        for i in remove_list:
            nonsilent_data.remove(i)


#         header = audio_bytes[:70]

#         user_talk_time = nonsilent_data

#         pTime = 1000
#         time_list = list(range(0,math.ceil(nonsilent_data[-1][-1]/pTime)))
#         time_per_1s = [[time_list[i]*pTime, time_list[i+1]*pTime]  for i in range(0,len(time_list)-1)]

#         audio_seg_u = []

#         for i in time_per_1s:
#             a = i[0] * 32 + 16000
#             b = i[1] * 32 + 16000
#             audio_seg_u.append(AudioSegment.from_raw(
#               io.BytesIO(header + audio_bytes[a:b]),
#               sample_width=2,
#               frame_rate=Freq,
#               channels=1
#             ))


#         volumes = []
#         for i in range(len(audio_seg_u)):
#             volumes.append(audio_seg_u[i].dBFS)

#         return volumes

        remove_list = []
        for i in nonsilent_data:
            if ((i[1] - i[0]) < 350):
                remove_list.append(i)
        for i in remove_list:
            nonsilent_data.remove(i)

        header = audio_bytes[:70]

        user_talk_time = nonsilent_data

        audio_seg_u = []

        for i in user_talk_time:
            a = i[0] * 32 + 16000
            b = i[1] * 32 + 16000
            audio_seg_u.append(AudioSegment.from_raw(
                io.BytesIO(header + audio_bytes[a:b]),
                sample_width=2,
                frame_rate=Freq,
                channels=1
            ))

        volumes = []
        for i in range(len(audio_seg_u)):
            volumes.append(audio_seg_u[i].dBFS)

        # volume class - 0: too quiet, 1: a little quiet, 2: appropriate, 3: a little loud, 4: too loud
        volume_class = []
        for i in volumes:
            volume_class.append(self.check_volume(i))

        return {'talk': user_talk_time, 'value': volumes, 'class': volume_class}

    def processE4(self, data):

        idx = data['idx']
        value = []
        iso = []
        xxx = []

        for i in idx:
            temp = [float(i) for i in data[data['idx'] == i]['value']]
            value.append(round(sum(temp) / len(temp), 4))
            t = float(i)
            iso.append(datetime.datetime.fromtimestamp(t).isoformat())
            xxx.append(datetime.datetime.fromtimestamp(t).isoformat()[14:19])

        pData = pd.DataFrame()
        pData['idx'] = idx
        pData['iso'] = iso
        pData['temp'] = xxx
        pData['value'] = value

        pattern = re.compile(r':\d\d')
        sec = []
        for i in pData['iso']:
            sec.append(re.findall(pattern, i)[1][1:])
        pData['sec'] = sec

        mValue = []
        xxx = sorted(list(set(xxx)))
        for i in xxx:
            xxxx = pData[pData['temp'] == i]['value']
            mValue.append(xxxx.mean())

        requestData = pd.DataFrame()
        requestData['timestamp'] = sorted(
            list(set([i[:19] for i in pData['iso']])))
        requestData['value'] = mValue

        pData = pData[['idx', 'iso', 'sec', 'value']]

        return pData, requestData
    
    def processEye(self, data):

        idx = data['idx']
        
        combineXvalue = []
        combineYvalue = []
        combineZvalue = []

        iso = []
        xxx = []
    
        for i in idx:
            combineX = [float(i) for i in data[data['idx'] == i][' combined_x']]
            combineY = [float(i) for i in data[data['idx'] == i][' combined_y']]
            combineZ = [float(i) for i in data[data['idx'] == i][' combined_z']]
            
            t = float(i)
            iso.append(datetime.datetime.fromtimestamp(t).isoformat())
            xxx.append(datetime.datetime.fromtimestamp(t).isoformat()[14:19])
            combineXvalue.append(round(sum(combineX) / len(combineX), 4))
            combineYvalue.append(round(sum(combineY) / len(combineY), 4))
            combineZvalue.append(round(sum(combineZ) / len(combineZ), 4))
            

        pData = pd.DataFrame()
        pData['idx'] = idx
        pData['iso'] = iso
        pData['temp'] = xxx
        pData['combine_x'] = combineXvalue
        pData['combine_y'] = combineYvalue
        pData['combine_z'] = combineZvalue

        pattern = re.compile(r':\d\d')
        sec = []
        for i in pData['iso']:
            sec.append(re.findall(pattern, i)[1][1:])
        pData['sec'] = sec

        combineXvalue = []
        combineYvalue = []
        combineZvalue = []

        xxx = sorted(list(set(xxx)))
        for i in xxx:
            xxxx = pData[pData['temp'] == i]['combine_x']
            combineXvalue.append(xxxx.mean())
        for i in xxx:
            xxxx = pData[pData['temp'] == i]['combine_y']
            combineYvalue.append(xxxx.mean())
        for i in xxx:
            xxxx = pData[pData['temp'] == i]['combine_z']
            combineZvalue.append(xxxx.mean())

        requestData = pd.DataFrame()
        requestData['timestamp'] = sorted(
            list(set([i[:19] for i in pData['iso']])))
        requestData['combine_x'] = combineXvalue
        requestData['combine_y'] = combineYvalue
        requestData['combine_z'] = combineZvalue

        pData = pData[['idx', 'iso', 'sec', 'combine_x', 'combine_y', 'combine_z']]

        return pData, requestData

    
    def processHmd(self, data):

        idx = data['idx']
        value = []
        iso = []
        xxx = []
        
        left_pos_x = []
        left_pos_y = []
        left_pos_z = []

        for i in idx:
            leftPosX = [float(i) for i in data[data['idx'] == i][' Left_pos.x']]
            leftPosY = [float(i) for i in data[data['idx'] == i][' Left_pos.y']]
            leftPosZ = [float(i) for i in data[data['idx'] == i][' Left_pos.z']]
            
            left_pos_x.append(round(sum(leftPosX) / len(leftPosX), 4))
            left_pos_y.append(round(sum(leftPosY) / len(leftPosY), 4))
            left_pos_z.append(round(sum(leftPosZ) / len(leftPosZ), 4))
            
            t = float(i)
            iso.append(datetime.datetime.fromtimestamp(t).isoformat())
            xxx.append(datetime.datetime.fromtimestamp(t).isoformat()[14:19])

        pData = pd.DataFrame()
        pData['idx'] = idx
        pData['iso'] = iso
        pData['temp'] = xxx
        pData['left_pos_x'] = left_pos_x
        pData['left_pos_y'] = left_pos_y
        pData['left_pos_z'] = left_pos_z

        pattern = re.compile(r':\d\d')
        sec = []
        for i in pData['iso']:
            sec.append(re.findall(pattern, i)[1][1:])
        pData['sec'] = sec

        left_pos_x = []
        left_pos_y = []
        left_pos_z = []
       
        xxx = sorted(list(set(xxx)))
        
        for i in xxx:
            xxxx = pData[pData['temp'] == i]['left_pos_x']
            left_pos_x.append(xxxx.mean())
        for i in xxx:
            xxxx = pData[pData['temp'] == i]['left_pos_y']
            left_pos_y.append(xxxx.mean())
        for i in xxx:
            xxxx = pData[pData['temp'] == i]['left_pos_z']
            left_pos_z.append(xxxx.mean())

        requestData = pd.DataFrame()
        requestData['timestamp'] = sorted(
            list(set([i[:19] for i in pData['iso']])))
        
        requestData['left_pos_x'] = left_pos_x
        requestData['left_pos_y'] = left_pos_y
        requestData['left_pos_z'] = left_pos_z


        pData = pData[['idx', 'iso', 'sec', 'left_pos_x', 'left_pos_y','left_pos_z']]

        return pData, requestData
    
    

    def makeRequestHmd(self, data):

        requestPosX = pd.DataFrame()
        requestPosY = pd.DataFrame()
        requestPosZ = pd.DataFrame()

        requestPosX['timestamp'] = data['timestamp']
        requestPosX['value'] = data[' Left_pos.x']

        requestPosY['timestamp'] = data['timestamp']
        requestPosY['value'] = data[' Left_pos.y']

        requestPosZ['timestamp'] = data['timestamp']
        requestPosZ['value'] = data[' Left_pos.z']

        return requestPosX, requestPosY, requestPosZ

    def makeRequestEyetracker(self, data):
        requestCombinedX = pd.DataFrame()
        requestCombinedY = pd.DataFrame()
        requestCombinedZ = pd.DataFrame()

        requestCombinedX['timestamp'] = data['timestamp']
        requestCombinedX['value'] = data[' combined_x']

        requestCombinedY['timestamp'] = data['timestamp']
        requestCombinedY['value'] = data[' combined_y']

        requestCombinedZ['timestamp'] = data['timestamp']
        requestCombinedZ['value'] = data[' combined_z']

        return requestCombinedX, requestCombinedY, requestCombinedZ

    def makeJson(self, data):
        p = {'granularity': 'secondly'}
        t = []
        for i, v in enumerate(data['timestamp']):
            t.append({'timestamp': str(v), 'value': data['value'][i]})
        p['series'] = t

        return p

    def json_default(self, value):
        if isinstance(value, datetime.date):
            return value.strftime('%Y-%m-%d')
        raise TypeError('not JSON serializable')

    def detect(self, endpoint, apikey, request_data):
        headers = {'Content-Type': 'application/json',
                   'Ocp-Apim-Subscription-Key': apikey}
        response = requests.post(endpoint, data=json.dumps(
            request_data, default=self.json_default), headers=headers)
        if response.status_code == 200:
            return json.loads(response.content.decode("utf-8"))
        else:
            print(response.status_code)
            raise Exception(response.text)

    def getAnomalyResult(self, data):
        result = self.detect(ANOMALY_DETECTOR_ENDPOINT, SUBSCRIPTION_KEY, data)
        result = {'expectedValues': result['expectedValues'], 'isAnomaly': result['isAnomaly'], 'isNegativeAnomaly': result['isNegativeAnomaly'],
                  'isPositiveAnomaly': result['isPositiveAnomaly'], 'upperMargins': result['upperMargins'], 'lowerMargins': result['lowerMargins'],
                  'timestamp': [x['timestamp'] for x in data['series']],
                  'value': [x['value'] for x in data['series']]}

        return {'value': result['value'], 'point': result['isAnomaly']}

    def makeChunk(self, data):
        chunk = []
        for i in range(0, len(data)-1):
            n = data[i]
            if n+1 < data[i+1]:
                chunk.append(data.index(n))
            elif i+2 == len(data):
                chunk.append(data.index(n))

        n = 0

        chunks = []
        for i in range(0, len(chunk)):
            chunks.append(data[n:chunk[i]+1])
            n = chunk[i] + 1

        return chunks

    def getSensorResult(self, sensor=None, voice=None):

        data = self.csv2df(sensor)

        anomalyVolume = None
        if voice is not None:
            volume = self.getVolume(voice)
#             volume = pd.DataFrame(columns=['idx','timestamp','value'])
#             volume['idx'] = [i for i in range(0, len(vo))]
#             volume['timestamp'] = [datetime.datetime.fromtimestamp(i).isoformat() for i in volume['idx']]
#             volume['value'] = [v for i, v in enumerate(vo)]
#             sendVolume = self.makeJson(volume)
#             volumeDict = self.getAnomalyResult(sendVolume)

#         for i, v in enumerate(volumeDict['value']):
#             if v < -60:
#                 volumeDict['value'][i] = volumeDict['expect'][i]

#         temp = volumeDict['value']
#         high = [i for i,v in enumerate(temp) if v > -15]
#         good = [i for i,v in enumerate(temp) if v <= -15 and v >= -35]

#         volumeDict['high'], volumeDict['good'] = self.getVoiceChunk(high, good)

        incol = []
        for i in data['unix']:
            incol.append(i - data['unix'][0])
        data['idx'] = incol
        self.df = data.copy()

        self.vrTime = data[['idx', 'vrTime']]

        eyeTracking = []
        vrTracking = []
        for i in sorted(list(set(data['idx']))):
            temp = [float(i) for i in data[data['idx'] == i][' PlayerSee']]
            if 1 in temp:
                eyeTracking.append(1)
            elif 2 in temp:
                eyeTracking.append(2)
            elif 1 not in temp:
                eyeTracking.append(0)

        for i in sorted(list(set(data['idx']))):
            temp = [float(i) for i in data[data['idx'] == i][' HaveToSee']]
            if 1 in temp:
                vrTracking.append(1)
            elif 2 in temp:
                vrTracking.append(2)
            elif 1 not in temp:
                vrTracking.append(0)

        hmd = data[['idx', ' Left_pos.x', ' Left_pos.y', ' Left_pos.z',
                    ' Left_rot.x', ' Left_rot.y', ' Left_rot.z']]
        e4Eda = data[[' EDA']]
        e4Bvp = data[[' BVP']]
        e4Tmp = data[[' TMP']]
        e4Ibi = data[[' IBI']]

        e4Bvp = pd.DataFrame(data=[i[8:].split()
                             for i in e4Bvp[' BVP']], columns=['unix', 'value'])
        e4Eda = pd.DataFrame(data=[i[8:].split()
                             for i in e4Eda[' EDA']], columns=['unix', 'value'])
        e4Tmp = pd.DataFrame(data=[i[16:].split()
                             for i in e4Tmp[' TMP']], columns=['unix', 'value'])
        e4Ibi = pd.DataFrame(data=[i[8:].split()
                             for i in e4Ibi[' IBI']], columns=['unix', 'value'])

        e4Bvp['idx'] = incol
        e4Eda['idx'] = incol
        e4Tmp['idx'] = incol
        e4Ibi['idx'] = incol

        e4Hr = e4Ibi.copy()
        a = np.full(len(e4Hr['value']), 60)
        e4Hr['value'] = a / np.array(e4Hr['value'], dtype=float)

        self.pBvp, self.requestBvp = self.processE4(e4Bvp)
        self.pEda, self.requestEda = self.processE4(e4Eda)
        self.pTmp, self.requestTmp = self.processE4(e4Tmp)
        self.pIbi, self.requestIbi = self.processE4(e4Ibi)
        self.pHr, self.requestHr = self.processE4(e4Hr)

        sendBvp = self.makeJson(self.requestBvp)
        sendEda = self.makeJson(self.requestEda)
        sendTmp = self.makeJson(self.requestTmp)
        sendIbi = self.makeJson(self.requestIbi)
        sendHr = self.makeJson(self.requestHr)

        timestamp = [x['timestamp'] for x in sendBvp['series']]
        bvpDict = self.getAnomalyResult(sendBvp)
        edaDict = self.getAnomalyResult(sendEda)
        tmpDict = self.getAnomalyResult(sendTmp)
        ibiDict = self.getAnomalyResult(sendIbi)
        hrDict = self.getAnomalyResult(sendHr)

        bvpPoint = [i for i, v in enumerate(bvpDict['point']) if v == True]
        edaPoint = [i for i, v in enumerate(edaDict['point']) if v == True]
        tmpPoint = [i for i, v in enumerate(tmpDict['point']) if v == True]
        ibiPoint = [i for i, v in enumerate(ibiDict['point']) if v == True]
        hrPoint = [i for i, v in enumerate(hrDict['point']) if v == True]

        anomalyPoints = sorted(
            list(set([*bvpPoint, *edaPoint, *tmpPoint, *ibiPoint, *hrPoint, ])))

        vrCoworker = [i for i, v in enumerate(vrTracking) if v == 2]
        vrGuest = [i for i, v in enumerate(vrTracking) if v == 1]
        vrNone = [i for i, v in enumerate(vrTracking) if v == 0]

        eyeCoworker = [i for i, v in enumerate(eyeTracking) if v == 2]
        eyeGuest = [i for i, v in enumerate(eyeTracking) if v == 1]
        eyeNone = [i for i, v in enumerate(eyeTracking) if v == 0]

        vrCoworker = self.makeChunk(vrCoworker)
        vrGuest = self.makeChunk(vrGuest)
        vrNone = self.makeChunk(vrNone)
        vrAll = [vrCoworker, vrGuest, vrNone]

        eyeCoworker = self.makeChunk(eyeCoworker)
        eyeGuest = self.makeChunk(eyeGuest)
        eyeNone = self.makeChunk(eyeNone)

        eyeCoworker = [[i[0], i[-1]] for i in eyeCoworker]
        eyeGuest = [[i[0], i[-1]] for i in eyeGuest]

        eyeAll = [eyeCoworker, eyeGuest, eyeNone]

        return {'timestamp': timestamp, 'eye': eyeAll, 'vr': vrAll,
                'bvp': bvpDict,
                'eda': edaDict,
                'volume': volume,
                'tmp': tmpDict,
                'ibi': ibiDict,
                'hr': hrDict,
                'anomaly': anomalyPoints}
    
    def getResult(self, sensor=None, voice=None):

        data = self.csv2df(sensor)

        anomalyVolume = None
        volume = []
        if voice is not None:
            volume = self.getVolume(voice)
        incol = []
        for i in data['unix']:
            incol.append(i - data['unix'][0])
        data['idx'] = incol
        self.df = data.copy()
        print("1")
        self.vrTime = data[['idx', 'vrTime']]

        hmd = data[['idx', ' Left_pos.x', ' Left_pos.y', ' Left_pos.z']]
        eye = data[['idx',' combined_x',' combined_y', ' combined_z',]]
        
        e4Eda = data[[' EDA']]
        e4Bvp = data[[' BVP']]
        e4Tmp = data[[' TMP']]
        e4Ibi = data[[' IBI']]

        e4Bvp = pd.DataFrame(data=[i[8:].split()
                             for i in e4Bvp[' BVP']], columns=['unix', 'value'])
        e4Eda = pd.DataFrame(data=[i[8:].split()
                             for i in e4Eda[' EDA']], columns=['unix', 'value'])
        e4Tmp = pd.DataFrame(data=[i[16:].split()
                             for i in e4Tmp[' TMP']], columns=['unix', 'value'])
        e4Ibi = pd.DataFrame(data=[i[8:].split()
                             for i in e4Ibi[' IBI']], columns=['unix', 'value'])

        e4Bvp['idx'] = incol
        e4Eda['idx'] = incol
        e4Tmp['idx'] = incol
        e4Ibi['idx'] = incol

        e4Hr = e4Ibi.copy()
        a = np.full(len(e4Hr['value']), 60)
        e4Hr['value'] = a / np.array(e4Hr['value'], dtype=float)

        self.pBvp, self.requestBvp = self.processE4(e4Bvp)
        self.pEda, self.requestEda = self.processE4(e4Eda)
        self.pTmp, self.requestTmp = self.processE4(e4Tmp)
        self.pIbi, self.requestIbi = self.processE4(e4Ibi)
        self.pHr, self.requestHr = self.processE4(e4Hr)
        print("2")
        self.pHmd, self.requestHmd = self.processHmd(hmd)
        self.pEye, self.requestEye = self.processEye(eye)

        sendBvp = self.makeJson(self.requestBvp)
        sendEda = self.makeJson(self.requestEda)
        sendTmp = self.makeJson(self.requestTmp)
        sendIbi = self.makeJson(self.requestIbi)
        sendHr = self.makeJson(self.requestHr)

        timestamp = [x['timestamp'] for x in sendBvp['series']]
        bvpDict = self.getAnomalyResult(sendBvp)
        edaDict = self.getAnomalyResult(sendEda)
        tmpDict = self.getAnomalyResult(sendTmp)
        ibiDict = self.getAnomalyResult(sendIbi)
        hrDict = self.getAnomalyResult(sendHr)

        bvpPoint = [i for i, v in enumerate(bvpDict['point']) if v == True]
        edaPoint = [i for i, v in enumerate(edaDict['point']) if v == True]
        tmpPoint = [i for i, v in enumerate(tmpDict['point']) if v == True]
        ibiPoint = [i for i, v in enumerate(ibiDict['point']) if v == True]
        hrPoint = [i for i, v in enumerate(hrDict['point']) if v == True]

        anomalyPoints = sorted(
            list(set([*bvpPoint, *edaPoint, *tmpPoint, *ibiPoint, *hrPoint, ])))
        print("3")
        sendHmd = {}
        sendHmd['x'] = self.requestHmd['left_pos_x'].to_list()
        sendHmd['y'] = self.requestHmd['left_pos_y'].to_list()
        sendHmd['z'] = self.requestHmd['left_pos_z'].to_list()
        
        sendEye = {}
        sendEye['x'] = self.requestEye['combine_x'].to_list()
        sendEye['y'] = self.requestEye['combine_y'].to_list()
        sendEye['z'] = self.requestEye['combine_z'].to_list()
        
        return {'timestamp': timestamp,
                'hmd': sendHmd,
                'eye': sendEye,
                'bvp': bvpDict,
                'eda': edaDict,
                'volume': volume,
                'tmp': tmpDict,
                'ibi': ibiDict,
                'hr': hrDict,
                'anomaly': anomalyPoints}

