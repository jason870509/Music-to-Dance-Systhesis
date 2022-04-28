import json
import numpy as np
import pandas as pd

if __name__ == '__main__':
    music_name = 'smooth_criminal'

    # JSON file
    f = open(f'output/{music_name}_data.json', "r")
    # Reading from file
    data = json.loads(f.read())

    data_length = len(data['start'])
    genres = data['genre']
    print(genres)
    starts = data['start']
    shift = data['shift']

    motion = []
    for i in range(data_length):
        genre = genres[i]
        start = starts[i]
        motion_data = np.load(f'data/keypoints3d_list/{genre}.npy')
        motion.append(motion_data[start:start+shift])

    motion = np.concatenate(motion)
    print(motion.shape)
    # 使用插值方式(取前後的中間)，先轉成dataframe型態，再使用interpolate算插值
    # df = pd.DataFrame(motion.reshape(motion.shape[0], -1))
    #
    # df = df.interpolate(method='linear', axis=0)
    # x = df.isnull().sum().sum()  # 計算nan的總數
    # print("NAN number: ", x)
    # # df.iloc[780][9*3+1] # 確認nan值被修正
    #
    # motion = df.to_numpy()
    # motion = motion.reshape(motion.shape[0], 17, 3)
    # print(motion.shape)
    keypoint_dict = {"keypoints": motion.tolist()}
    # # print(keypoint_dict.keys())
    # df = pd.DataFrame.from_dict(keypoint_dict, orient='index')
    #
    # # Convert to json values
    # json_df = df.to_json(orient='values', date_format='iso', date_unit='s')

    with open(f'output/{music_name}_motion_data.json', 'w') as js_file:
        # js_file.write(json_df)
        json.dump(motion.tolist(), js_file, indent=4)


