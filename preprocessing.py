'''
preprocessing.py : 전처리와 관련된 파이썬 파일입니다.
'''
from pyannote.audio import Model, Inference
import librosa
import torch

'''
test_librosa : librosa 전처리 코드
'''
def test_librosa():
    # waveform : 실수형 ndarray -> 시각화하는데 사용
    # sr : sampling_rate
    waveform, sr = librosa.load('dataset/raw_data/2_1373G2A4_1447G2A5_T1_2D07T0333C000478_005449.wav')

    # 데이터 길이 / sampling rate -> 음성 파일 시간
    print(f'음성 시간 : {len(waveform) // sr // 3600}h: {len(waveform) // sr // 60}m: {len(waveform) // sr % 60}s')




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