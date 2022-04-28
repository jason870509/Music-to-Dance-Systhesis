import json
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # JSON file
    f = open('output/hiphop_data.json', "r")
    # Reading from file
    data = json.loads(f.read())

    data_length = len(data['genre'])
    genres = data['genre']
    starts = data['start']
    shift = 180

    motion = []
    for i in range(1):
        print(genres[0])
        genre = genres[0]
        start = starts[0]
        motion_data = np.load(f'data/keypoints3d_list/{genre}.npy')
        motion.append(motion_data[start:start+360])

    motion = np.concatenate(motion)
    print(motion.shape)

    keypoint_dict = {"keypoints": motion.tolist()}
    # # print(keypoint_dict.keys())
    # df = pd.DataFrame.from_dict(keypoint_dict, orient='index')
    #
    # # Convert to json values
    # json_df = df.to_json(orient='values', date_format='iso', date_unit='s')

    with open('output/origin_data2.json', 'w') as js_file:
        # js_file.write(json_df)
        json.dump(motion.tolist(), js_file, indent=4)


