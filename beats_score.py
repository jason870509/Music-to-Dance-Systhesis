from extract_music import *
import torch
import json
import numpy as np
import time
from crnn import CRNNNetwork
import scipy.signal as scisignal


def alignment_score(music_beats, motion_beats, sigma=3):
    """Calculate alignment score between music and motion."""
    if motion_beats.sum() == 0:
        return 0.0
    # beats 所以有位置陣列
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]

    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        # ind: dists最小值的index
        ind = np.argmin(dists)
        score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        score_all.append(score)
    return sum(score_all) / len(score_all)

def motion_peak_onehot(joints):
    # Calculate velocity.
    # v = 距離 / t(相等) ==> v = 距離
    velocity = np.zeros_like(joints, dtype=np.float32)
    # 每一個frame的joints座標都減去前一個frame的joints座標，得到每個joints中(x, y, z)的移動距離
    velocity[1:] = joints[1:] - joints[:-1] # shape = (seq_len, 24, 3)
    # 將每個frame的(x, y, z)移動的距離平方相加開根號=總移動距離(joints移動距離)
    velocity_norms = np.linalg.norm(velocity, axis=2) # shape = (seq_len, 24)
    # 將全部joints的移動距離相加
    envelope = np.sum(velocity_norms, axis=1)  # shape = (seq_len,)
    # print(envelope)
    # Find local minima in velocity -- beats
    # 求數組的極小值
    # order: 兩側使用多少點進行比較
    # np.less 保存數組最小數值
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10) # 10 for 60FPS
    # print(peak_idxs)
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1
    # # Second-derivative of the velocity shows the energy of the beats
    # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
    # # optimize peaks
    # peak_onehot[peak_energy<0.001] = 0
    return peak_onehot


def predict_music(music_path):
    melgrams = np.load(music_path)
    music_beats = []
    for i in melgrams:
        music_beats.append(i[:, -1])

    tensor_melgrams = torch.Tensor(melgrams)
    crnn = CRNNNetwork()
    weight = torch.load('model/crnn.pth')
    crnn.load_state_dict(weight)
    h_n, h_c = torch.randn(2, 32, 32), torch.randn(2, 32, 32)

    crnn.eval()
    predict_genres = []
    with torch.no_grad():
        for b_x in tensor_melgrams:
            b_x = b_x
            b_x = torch.reshape(b_x, (1, 1, 180, 18))

            test_output, _, _ = crnn(b_x, h_n, h_c)
            pred_y = test_output.argmax(1)
            predict_genres.append(motion_labels[pred_y.item()])

    return  predict_genres, music_beats


if __name__ == '__main__':
    music_name = 'hip_hop'
    melgrams = np.load(f'data/input_music/npy/{music_name}.npy')
    music_beats = []
    for i in melgrams:
        music_beats.append(i[:, -1])
    music_beats = np.concatenate(music_beats)
    # predict_genres, music_beats= predict_music(music_path=f'data/input_music/npy/{music_name}.npy')
    # JSON file
    f = open(f'output/{music_name}_data.json', "r")
    # Reading from file
    data = json.loads(f.read())

    data_length = len(data['genre'])
    genres = data['genre']
    starts = data['start']
    shift = 180

    motion = []
    for i in range(data_length):
        genre = genres[i]
        start = starts[i]
        motion_data = np.load(f'data/keypoints3d_list/{genre}.npy')
        motion.append(motion_data[start:start+shift])

    motion = np.concatenate(motion)
    motion_beats = motion_peak_onehot(motion)

    score = alignment_score(music_beats, motion_beats, sigma=3)
    print("generate dance score:", score)

    motion = []
    for i in range(1):
        genre = genres[i]
        start = starts[i]
        motion_data = np.load(f'data/keypoints3d_list/{genre}.npy')
        motion.append(motion_data[start:start+1800])

    motion = np.concatenate(motion)
    motion_beats = motion_peak_onehot(motion)

    score = alignment_score(music_beats, motion_beats, sigma=3)
    print("origin dance score:", score)




