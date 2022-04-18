#!/usr/bin/env python
# coding: utf-8

# In[122]:


from urllib import request
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
# from pydub.silence import split_on_silence
import io
import collections
import contextlib
import sys
import wave
import webrtcvad
# import os
from scipy.io import wavfile
# import scipy.io
# import ssl


# In[617]:


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):

    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    percent_thresh = 0.95  # default: 0.9

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > percent_thresh * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > percent_thresh * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def check_conflict(guide_time, voice_chunk):
    if (voice_chunk[1] < guide_time[0]) or (voice_chunk[0] > guide_time[1]):
        return None
    else:
        if voice_chunk[0] < guide_time[0]:
            if voice_chunk[1] < guide_time[1]:
                return ([guide_time[0], voice_chunk[1]],
                        voice_chunk[1] - guide_time[0])
            elif voice_chunk[1] >= guide_time[1]:
                return guide_time, guide_time[1] - guide_time[0]
        else:
            if voice_chunk[1] < guide_time[1]:
                return voice_chunk, voice_chunk[1] - voice_chunk[0]
            elif voice_chunk[1] >= guide_time[1]:
                return ([voice_chunk[0], guide_time[1]],
                        guide_time[1] - voice_chunk[0])


def check_hesitate(guide_time, voice_chunk):
    if voice_chunk[0]-guide_time[1] >= 3000:
        return ([guide_time[1], voice_chunk[0]], voice_chunk[0]-guide_time[1])
    else:
        return None


def check_volume(vol):
    threshold = [-29, -24.6, -20.2, -15.8]

    if vol <= threshold[0]:
        return 0
    elif vol <= threshold[1]:
        return 1
    elif vol <= threshold[2]:
        return 2
    elif vol <= threshold[3]:
        return 3
    else:
        return 4


def make_voice_segment(audio, sample_rate):
    vad = webrtcvad.Vad(0)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 500, vad, frames)

    concataudio = [segment for segment in segments]
    joinedaudio = b"".join(concataudio)
    seg = AudioSegment.from_raw(
        io.BytesIO(joinedaudio),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1
    )
    return seg, seg.dBFS


def voice_py():

    # scene_timestamp = [1615914137 * 1000 * 32, 1615914182 * 1000 * 32, 1615914200 * 1000 * 32, 1615914239 * 1000 * 32]
    # scene_bytes = []
    # for i in range(len(scene_timestamp) - 1):
    #     scene_bytes.append([scene_timestamp[i] - scene_timestamp[0], scene_timestamp[i + 1] - scene_timestamp[0]])

    # scene_time = [[scene_bytes[0][0], int(scene_bytes[0][1] / 32)],
    #               [int(scene_bytes[1][0] / 32), int(scene_bytes[1][1] / 32)],
    #               [int(scene_bytes[2][0] / 32), int(scene_bytes[2][1] / 32)]]

    # # path from URL
    # url = "https://dt1amnyxy57si.cloudfront.net/audios/0317_Audio.wav"
    # Freq = 16000
    # context = ssl._create_unverified_context()
    # audio_bytes = request.urlopen(url,context=context).read()

    # #path from local
    # audio_bytes = wave.open("/content/20220308_210750_voice.wav",'rb')
    # audio_bytes = audio_bytes.readframes(14400000)

    # Convert wav to audio_segment
    audio_segment = AudioSegment.from_raw(
        io.BytesIO(audio_bytes),
        sample_width=2,
        frame_rate=Freq,
        channels=1
    )

    # Normalize audio_segment to -20dBFS
    # normalized_sound = match_target_amplitude(audio_segment, -10.0)
    # print("length of audio_segment = {} ms".format(len(audio_segment)))

    # Receive detected non-silent chunks, which in our case would be spoken words.
    nonsilent_data = detect_nonsilent(
        audio_segment,
        min_silence_len=600,
        silence_thresh=-37,
        seek_step=1
    )

    # sentence_chunks=[[0 for col in range(2)] for row in range(len(nonsilent_data))]

    # print("\n[Start, Stop](ms): voice")
    # for sentences in nonsilent_data:
    #     print([sentence for sentence in sentences])
    remove_list = []
    for i in nonsilent_data:
        if ((i[1] - i[0]) < 350):
            remove_list.append(i)
    for i in remove_list:
        nonsilent_data.remove(i)

    # In[625]:

    header = audio_bytes[:70]

    user_talk_time = nonsilent_data

    time_list = list(range(0, math.ceil(nonsilent_data[-1][-1]/1000)+5))
    # print("Time list:", time_list, '\n')

    time_per_1s = []
    for i in range(len(time_list)-1):
        time_per_1s.append([time_list[i]*1000, time_list[i+1]*1000])
        # time_per_1s[i][1] = time_list[i+1]
    # print("Time per 1s", time_per_1s, '\n')

    # vr_talk_time = [[26546, 30266], [37494, 40004], [65256, 67436], [87211, 89482]]

    # In[626]:

    # audio_scene = []
    # for i in range(len(scene_bytes)):
    #     audio_scene.append(AudioSegment.from_raw(
    #         io.BytesIO(header + audio_bytes[scene_bytes[i][0] + 16000:scene_bytes[i][1] + 16000]),
    #         sample_width=2,
    #         frame_rate=Freq,
    #         channels=1
    #     ))

    audio_seg_u = []

    # for i in user_talk_time:
    #     a = i[0] * 32 + 16000
    #     b = i[1] * 32 + 16000
    #     audio_seg_u.append(AudioSegment.from_raw(
    #         io.BytesIO(header + audio_bytes[a:b]),
    #         sample_width=2,
    #         frame_rate=Freq,
    #         channels=1
    #     ))
    for i in time_per_1s:
        a = i[0] * 32 + 16000
        b = i[1] * 32 + 16000
        audio_seg_u.append(AudioSegment.from_raw(
            io.BytesIO(header + audio_bytes[a:b]),
            sample_width=2,
            frame_rate=Freq,
            channels=1
        ))
    # In[635]:

    volumes = []
    for i in range(len(audio_seg_u)):
        volumes.append(audio_seg_u[i].dBFS)

    # In[637]:

    # volume class - 0: too quiet, 1: a little quiet, 2: appropriate, 3: a little loud, 4: too loud
    # volume_class = []
    # for i in volumes:
    #     volume_class.append(check_volume(i))

    # In[639]:

    # conflicts = []
    # for i in user_talk_time:
    #     for j in vr_talk_time:
    #         if check_conflict(j, i) != None:
    #             conflicts.append(check_conflict(j, i))
    #         else:
    #             continue

    # In[641]:

    # hesitate = []
    # if check_hesitate(vr_talk_time[0], user_talk_time[1]) != None:
    #     hesitate.append(check_hesitate(vr_talk_time[0], user_talk_time[1]))
    # if check_hesitate(vr_talk_time[3], user_talk_time[3]) != None:
    #     hesitate.append(check_hesitate(vr_talk_time[3], user_talk_time[3]))

    volume_graph_1s = []
    for i in range(int((len(audio_bytes) - 70) / 32000)):
        volume_graph_1s.append((AudioSegment.from_raw(
            io.BytesIO(header + audio_bytes[i + 70:i + 70 + 32000]),
            sample_width=2,
            frame_rate=Freq,
            channels=1
        )).dBFS)

    print('[ms]', user_talk_time, '| \n', volumes)
    # plt.plot(volumes)


voice_py()
