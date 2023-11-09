import os

data_dir = '/work/yw410445/dataset/sequences/'

sequences = sorted(os.listdir(data_dir))

framesCnt = 0
for sequence in sequences[:11]:
    frames = os.listdir(os.path.join(data_dir, sequence, 'labels'))
    framesCnt += len(frames)
    print(f'{sequence}: {len(frames)}')

print(framesCnt)