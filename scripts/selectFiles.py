import os
import numpy as np
import shutil

data_dir = '/work/yw410445/training_data'
out_dir = '/work/yw410445/training_data_reduced'
num_input_features = 3

skip_training = 6
skip_validation = 5

def copy_with_skip(path, step = 1):
    global num_input_features, out_dir

    sequences = os.listdir(path)
    for sequence in sorted(sequences):
        print(f'Start with sequence {sequence}...')
        path_gnd_labels = os.path.join(path, sequence, 'gnd_labels')
        path_reduced_velo = os.path.join(path, sequence, 'reduced_velo')
        frames = os.listdir(path_gnd_labels)

        out_dir_seq = os.path.join(out_dir, sequence)
        out_data = os.path.join(out_dir_seq, 'reduced_velo')
        out_labels = os.path.join(out_dir_seq, 'gnd_labels')
        os.makedirs(out_data, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

        for i, frame in enumerate(sorted(frames)):
            if i % step == 0:
                data = np.load(os.path.join(path_reduced_velo, frame))[:,:num_input_features]
                np.save(os.path.join(out_data, f'{i:06d}.npy'), data)
                shutil.copyfile(os.path.join(path_gnd_labels, frame), os.path.join(out_labels, f'{i:06d}.npy'))

if __name__ == '__main__':
    copy_with_skip(os.path.join(data_dir, 'training'), skip_training)
    copy_with_skip(os.path.join(data_dir, 'validation'), skip_validation)