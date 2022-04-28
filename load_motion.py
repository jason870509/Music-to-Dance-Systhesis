from extract_music import *
import torch
import json
import numpy as np
import time
from crnn import CRNNNetwork
import scipy.signal as scisignal
motion_labels = 'gBR gHO gJB gJS gKR gLH gLO gMH gPO gWA'.split()

motion_list = {
    "genre": [],
    "shift": 0,
    "score": [],
    "start": [],
}

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

def motion_compare_distance(pre_joints, joints):

    velocity = pre_joints[-1:] - joints[:1] # shape = (seq_len, 24, 3)
    # velocity = abs(velocity.reshape(75))
    # velocity[:3] = velocity[:3] * 100
    # print('velocity:', velocity, velocity.shape)
    # print(velocity.shape)
    velocity_norms = np.linalg.norm(velocity, axis=2) # shape = (seq_len, 24)
    # print('velocity_norms: ', velocity_norms, velocity_norms.shape)
    # 將全部joints的移動距離相加
    # print(velocity.shape)
    envelope = np.sum(velocity_norms, axis=1)  # shape = (seq_len,)
    # print('envelope: ', envelope, envelope.shape)
    return envelope[0]

def motion_compare_angle(pre_joints, joints):

    velocity = pre_joints[-1:] - joints[:1] # shape = (seq_len, 24, 3)
    velocity = abs(velocity.reshape(75))
    # velocity[:3] = velocity[:3] * 100
    # print('velocity:', velocity, velocity.shape)
    # print(velocity.shape)
    # velocity_norms = np.linalg.norm(velocity, axis=2) # shape = (seq_len, 24)
    # print('velocity_norms: ', velocity_norms, velocity_norms.shape)
    # 將全部joints的移動距離相加
    # print(velocity.shape)
    envelope = np.sum(velocity, axis=0)  # shape = (seq_len,)
    # print('envelope: ', envelope, envelope.shape)
    print(envelope)
    return envelope

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

def decide_music_genre(predict):
    max_genre = np.zeros(10)
    for i in predict:
        index = motion_labels.index(i)
        max_genre[index] += 1
    return motion_labels[np.argmax(max_genre)]

if __name__ == '__main__':
    music_name = 'smooth_criminal'
    # predict_genres, music_beats= predict_music(music_path=f'data/input_music/npy/{music_name}.npy')
    # music_genre = decide_music_genre(predict_genres)
    # print(music_genre)
    tempo = np.load(f'data/input_music/npy/{music_name}_tempo.npy')
    beats = np.load(f'data/input_music/npy/{music_name}_beats.npy')
    genres = np.load(f'data/input_music/npy/{music_name}_genre.npy')
    print(genres)
    beat_start = 0

    for i in beats:
        if i == 1:
            break
        beat_start += 1
    print("beat_start: ", beat_start)
    beats = beats[beat_start:]
    # print(beats[:230])
    # print(beats)
    # print(tempo, beats, beats.shape)
    beat_time = 60 / tempo
    beats_time = 8 * beat_time
    frame_time = 1 / 60
    frames_time = beats_time / frame_time
    frames_time = math.ceil(frames_time)
    print("frames_time: ", frames_time)
    total_step = math.floor(beats.shape[0] // frames_time) - 1
    print(total_step )
    motion_list['shift'] = frames_time
    # test load npy
    shift = 60
    # step = 0
    pre_motion = []

    # for g in predict_genres:
    #     break


    for step in range(total_step):
        beat_scores = []
        motion_distances = []
        motion_seq_index = 0
        now_time = frames_time * step / 60
        g = motion_labels[genres[math.floor(now_time / 3)]]
        print(g)
        keypoints3d = np.load(f'data/keypoints3d_list/{g}.npy')
        motion = np.load(f'data/motion_origin_list/{g}.npy')
        motion_seq = np.load(f'data/motion_origin_list/{g}_seq.npy')

        for i in range((keypoints3d.shape[0]- frames_time)// shift):

            start = i * shift
            # print(i, keypoints3d.shape[0]-181)
            if start in pre_motion:
                print("Same with pre motion.")
                motion_distances.append(10000)
                beat_scores.append(0)
                continue
            # if start >= motion_seq[motion_seq_index]:
            #     motion_seq_index += 1
            # if start+180 >= motion_seq[motion_seq_index]:
            #     print(start,  motion_seq[motion_seq_index])
            #     # print("start:", start + 180, i)
            #     continue

            end = start + frames_time
            keypoints3d_data = keypoints3d[start:end]
            motion_data = motion[start:end]

            motion_beats = motion_peak_onehot(keypoints3d_data)
            music_start = step * frames_time
            music_end = music_start + frames_time
            score = alignment_score(beats[music_start:music_end], motion_beats, sigma=3)

            if step != 0:
                motion_distance = motion_compare_distance(keypoints3d[pre_motion[-1]:pre_motion[-1]+180],
                                                              keypoints3d_data)
                # motion_distance = motion_compare_angle(motion[pre_motion[-1]:pre_motion[-1] + 180],
                #                                           motion_data)
                if np.isnan(motion_distance):
                    motion_distances.append(10000)
                else:
                    motion_distances.append(motion_distance)

            # print(score)
            beat_scores.append(score)

        print(len(beat_scores))
        if step > 0:
            max_beat = 0
            min_distance = 100000
            min_index = -1
            for index, j in enumerate(beat_scores):
                # print(index, motion_distances[index])
                if max_beat < j :
                    if index != 0:
                        if min_distance > motion_distances[index]:
                            max_beat = j
                            min_distance = motion_distances[index]
                            min_index = index
                    else:
                        max_beat = j
                        min_distance = motion_distances[index]
                        min_index = index
            print(max_beat, min_distance, min_index)
            beat_score = min_index
        else:
            beat_score = np.argmax(beat_scores)

        # step += 1
        motion_list['genre'].append(g)
        motion_list['score'].append(beat_scores[beat_score])
        motion_list['start'].append(int(beat_score * shift))
        print(beat_score * shift, beat_scores[beat_score])

        if len(pre_motion) == 2:
            pre_motion.pop(0)
        pre_motion.append(beat_score * shift)


    print(len(motion_list['score']), len(motion_list['start']))
    with open(f'output/{music_name}_data.json', "w") as fp:
        json.dump(motion_list, fp, indent=4)
