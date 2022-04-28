import json
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # JSON file
    music = 'scream'
    f = open(f'output/{music}_data.json', "r")
    # Reading from file
    data = json.loads(f.read())

    data_length = len(data['genre'])
    genres = data['genre']
    starts = data['start']
    shift = data['shift']

    motion = []
    for i in range(data_length):
        genre = genres[i]
        start = starts[i]
        motion_data = np.load(f'data/motion_list/{genre}.npy')
        motion.append(motion_data[start:start+shift])

    motion = np.concatenate(motion)
    motion = motion[np.newaxis,:,:]
    print(motion.shape)
    np.save('data/gen_motion.npy', motion)

