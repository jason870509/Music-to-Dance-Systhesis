import numpy as np
import math
import librosa
import os

DATASET_PATH = "./music/music"
SAMPLE_RATE = 60 * 512
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def extract_music(music_path, num_mfcc=13, hop_length=512, time_clip=3):
    melgrams = np.zeros((0, 180, 18))
    samples_per_segment = time_clip * SAMPLE_RATE  # int(SAMPLES_PER_TRACK / num_segments) # 661500/num_segments
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)  # 180( math.ceil無條件進位 )

    signal, sample_rate = librosa.load(music_path, sr=SAMPLE_RATE)
    num_segments = signal.shape[0] // (time_clip * sample_rate)

    # 梅爾頻率倒譜係數
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=15).T  # (seq_len, n_mfcc)
    mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min())
    mfcc = mfcc * (1 - (-1)) + (-1)
    # mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min())
    # mfcc = mfcc * (1 - (-1)) + (-1)
    envelope = librosa.onset.onset_strength(y=signal, sr=sample_rate)  # (seq_len,)
    envelope = (envelope - envelope.min()) / (envelope.max() - envelope.min())
    envelope = envelope * (1 - 0) + 0
    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=sample_rate, hop_length=hop_length)
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0  # (seq_len,)
    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope, sr=sample_rate, hop_length=hop_length)
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)

    audio_feature = np.concatenate([
        envelope[:, None], mfcc, peak_onehot[:, None], beat_onehot[:, None]
    ], axis=-1)[:-1]  # (seq_len, 18)
    print( audio_feature.shape )
    # process all segments of audio file(分割音頻)
    for d in range(num_segments):  # num_segments
        # calculate start and finish sample for current segment
        start = num_mfcc_vectors_per_segment * d
        finish = start + num_mfcc_vectors_per_segment

        # if len(audio_feature) == num_mfcc_vectors_per_segment:
        # audio_feature = audio_feature[np.newaxis, :, :]
        melgrams = np.concatenate((melgrams, audio_feature[np.newaxis, start:finish]), axis=0)
        print("segment:{}".format(d + 1))
        # print( start, finish )
        # start = samples_per_segment * d
        # finish = start + samples_per_segment
    #     y = signal[start:finish]
        # # 計算光譜通量起始強度包絡(envelope)
        # envelope = librosa.onset.onset_strength(y=y, sr=sample_rate)  # (seq_len,)
        # envelope = (envelope - envelope.min()) / (envelope.max() - envelope.min())
        # envelope = envelope * (1 - 0) + 0
        # if np.isnan(envelope.max()):
        #     print("envelope is nan")
        #     continue
        #
        # # 通過在起始強度包絡中選取峰值來定位音符起始事件
        # peak_idxs = librosa.onset.onset_detect(
        #     onset_envelope=envelope.flatten(), sr=sample_rate, hop_length=hop_length)
        # peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        # peak_onehot[peak_idxs] = 1.0  # (seq_len,)
        # # 偵測節拍
        # tempo, beat_idxs = librosa.beat.beat_track(
        #     onset_envelope=envelope, sr=sample_rate, hop_length=hop_length, tightness=100)
        # beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        # beat_onehot[beat_idxs] = 1.0  # (seq_len,)

        # audio_feature = np.concatenate([
        #     envelope[:, None], mfcc, peak_onehot[:, None], beat_onehot[:, None]
        # ], axis=-1)[:-1]  # (seq_len, 18)

        # if len(audio_feature) == num_mfcc_vectors_per_segment:
        #     audio_feature = audio_feature[np.newaxis, :, :]
        #     melgrams = np.concatenate((melgrams, audio_feature), axis=0)
        #     print("segment:{}".format(d + 1))

    return melgrams, tempo, beat_onehot


if __name__ == '__main__':
    music_name = 'smooth_criminal'
    melgrams, tempo, beats = extract_music(music_path=f'data/input_music/wav/{music_name}.wav')

    print(melgrams.shape, tempo)
    # print(melgrams)
    np.save(f'data/input_music/npy/{music_name}.npy', melgrams)
    np.save(f'data/input_music/npy/{music_name}_tempo.npy', tempo)
    np.save(f'data/input_music/npy/{music_name}_beats.npy', beats)