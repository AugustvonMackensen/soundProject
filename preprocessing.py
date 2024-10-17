'''
preprocessing.py : 전처리와 관련된 파이썬 파일입니다.
'''
import numpy as np
from pyannote.audio import Model, Inference
import matplotlib.pyplot as plt
import librosa
from librosa.feature import melspectrogram
import torch

'''
test_librosa : librosa 전처리 코드
'''
def test_librosa(file_path):
    # waveform : 실수형 ndarray -> 시각화하는데 사용
    # sr : sampling_rate
    waveform, sr = librosa.load('dataset/raw_data/2_1373G2A4_1447G2A5_T1_2D07T0333C000478_005449.wav')

    # 데이터 길이 / sampling rate -> 음성 파일 시간
    print(f'음성 시간 : {len(waveform) // sr // 3600}h: {len(waveform) // sr // 60}m: {len(waveform) // sr % 60}s')

    # 멜스토그램 만들기
    # win_length : 자를때 자를 작은 조각의 크기(기본이 25ms)
    # n_fft : win_length 크기로 잘린 음성의 작은 조각은 0으로 패딩되어서 n_fft로 크기가 맞춰짐. (일반적으로, 속도를 위해서 2^n으로 설정해야 함, win_length보다 크거나 같아야 함)
    # hop_length : 자르는 간격
    # librosa.feature.melspectogram : melspectogram ->

    win_length = int(np.ceil(0.025 * len(waveform)))
    hop_length = int(np.ceil(0.001 * len(waveform)))




def melstogram(waveform, sr, win_length, hop_length):
    s = melspectrogram(y=waveform, sr=sr, n_fft=1024, win_length=win_length, hop_length=hop_length)
    fig, ax = plt.subplots()
    s_db = librosa.power_to_db(s, ref=np.max)
    img = librosa.display.specshow(s_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)





# def test_rt_speaker():
# test code 404 발생으로 주석처리 해놨습니다!
#     model = Model.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         use_auth_token="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")
#
#     # gpu에 적용
#     model.to('mps')
#
#     # apply pretrained pipeline
#
#     inference = Inference(model, step=2.5)
#     output = inference("/Users/a-25/Documents/soundProject/dataset/raw_data/2_1373G2A4_1447G2A5_T1_2D07T0333C000478_005449.wav")
#
#     # print the result
#     for turn, _, speaker in output.itertracks(yield_label=True):
#         print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
#     # start=0.2s stop=1.5s speaker_0
#     # start=1.8s stop=3.9s speaker_1
#     # start=4.2s stop=5.7s speaker_0
#     # ...

if __name__ == '__main__':
    test_librosa()