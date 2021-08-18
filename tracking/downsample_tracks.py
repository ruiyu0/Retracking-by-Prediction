import os
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Downsample tracks')
    parser.add_argument('--save_dir', default="../output", type=str)
    parser.add_argument('--dataset', default="little3", type=str)
    parser.add_argument('--fps', default=2.5, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    trk_all = np.loadtxt('../output/{}_sort.txt'.format(args.dataset),delimiter=',')
    trk_all = trk_all[:,:6]

    # Normalize the IDs
    start_id = trk_all[:,1].min()
    trk_all[:,1] = trk_all[:,1] - start_id + 1
    unique_labels = np.unique(trk_all[:, 1])
    num_ids = len(unique_labels)

    # Interpolate the tracks
    trk_interp = []
    for i in range(num_ids):
        idx = trk_all[:,1] == unique_labels[i]
        trk_i = trk_all[idx, :]
        trk_i = trk_i[trk_i[:,0].argsort()] # sort by first column (frame) in ascending order

        # Ignore the id with only one frame
        if trk_i.shape[0] == 1: 
            continue
        frames = np.arange(trk_i[0,0], trk_i[-1,0]+1, 1) # continous frames to be interpolated
        new_trk_i = np.zeros((len(frames),6))
        new_trk_i[:,0] = frames
        new_trk_i[:,1] = unique_labels[i]
        new_trk_i[:,2] = np.interp(frames, trk_i[:,0], trk_i[:,2])
        new_trk_i[:,3] = np.interp(frames, trk_i[:,0], trk_i[:,3])
        new_trk_i[:,4] = np.interp(frames, trk_i[:,0], trk_i[:,4])
        new_trk_i[:,5] = np.interp(frames, trk_i[:,0], trk_i[:,5])
        trk_interp.append(new_trk_i)

    trk_all = np.concatenate(trk_interp, axis=0)

    # Downsample the tracks
    idx = trk_all[:, 0]%(30//args.fps)==0
    new_trk_all = trk_all[idx, :]

    save_file_path = os.path.join(args.save_dir, '{}_sort_fps{}.txt'.format(args.dataset, args.fps))
    np.savetxt(save_file_path, new_trk_all, fmt='%1.3f', delimiter=',')

    print('Finished downsampling tracks to {} FPS'.format(args.fps))
