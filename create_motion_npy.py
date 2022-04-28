import os
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R



if __name__ == '__main__':
    motion_dir= 'data/motions/'
    motion_labels = 'gBR gHO gJB gJS gKR gLH gLO gMH gPO gWA'.split()

    # create genres data (.npy)
    for g in motion_labels:
        motion_list = []
        sum_sequence = []
        sum_seq = 0
        j = 0
        for i in os.listdir(f'data/motions/{g}/'):
            path = f'data/motions/{g}/' + i
            # print(path)
            with open(path, 'rb') as f:
                data = pickle.load(f)
            # keypoint_data = data['keypoints3d']

            # train data 消 nan 值
            # 使用插值方式(取前後的中間)，先轉成dataframe型態，再使用interpolate算插值
            # df = pd.DataFrame(keypoint_data.reshape(keypoint_data.shape[0], -1))
            #
            # df = df.interpolate(method='linear', axis=0)
            # x = df.isnull().sum().sum()  # 計算nan的總數
            # print("NAN number: ", x)
            # df.iloc[780][9*3+1] # 確認nan值被修正

            # keypoint_data = df.to_numpy()
            # keypoint_data = keypoint_data.reshape(keypoint_data.shape[0], 17, 3)
            # print(keypoint_data.shape)
            # 12.5s
            smpl_poses = data['smpl_poses']  # (N, 24, 3)
            smpl_scaling = data['smpl_scaling']  # (1,)
            smpl_trans = data['smpl_trans']  # (N, 3)
            smpl_trans /= smpl_scaling

            # smpl_poses = R.from_rotvec(
            #     smpl_poses.reshape(-1, 3)).as_matrix().reshape(smpl_poses.shape[0], -1)
            smpl_motion = np.concatenate([smpl_trans, smpl_poses], axis=-1)
            # smpl_motion = np.pad(smpl_motion, [[0, 0], [6, 0]])
            motion_list.append(smpl_motion)
            sum_seq += smpl_motion.shape[0]
            sum_sequence.append(sum_seq)
            j += 1
            # if len(motion_list) == 2: break
        motion_list = np.concatenate(motion_list, axis=0).reshape(-1, 25, 3)
        print(motion_list.shape, sum_seq, g)
        # np.save(f'data/motion_origin_list/{g}.npy', motion_list)
        print(np.array(sum_sequence))
        np.save(f'data/motion_origin_list/{g}_seq.npy', np.array(sum_sequence))
        # break
    # test load npy
    # for g in motion_labels:
    #     data = np.load(f'data/keypoints3d_list/{g}.npy')
    #     print(data.shape)


