import torch
import os
import numpy as np
import argparse
from utils import relative_to_abs, convert_pixel_to_world, convert_world_to_pixel

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='../output', type=str)
parser.add_argument('--model_dir', default='saved_models', type=str)
parser.add_argument('--homo_dir', default='../data/homography', type=str)
parser.add_argument('--train_dataset', default='trajnet_train', type=str)
parser.add_argument('--test_dataset', default='little3', type=str)
parser.add_argument('--obs_len', default=4, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--fps', default=2.5, type=float)
args = parser.parse_args()


def bbox_predict(vlstm_gt, all_obs, args):
    obs_len = args.obs_len
    pred_len = args.pred_len
    seq_len = args.obs_len + args.pred_len

    all_pred = np.zeros((pred_len, all_obs.shape[1], all_obs.shape[2]))
    obs_center = all_obs[:,:,2:4]
    obs_traj = torch.from_numpy(obs_center).type(torch.float)
    obs_traj_rel = torch.zeros(obs_traj.shape, dtype=torch.float)
    obs_traj_rel[1:,:,:] = obs_traj[1:,:,:] - obs_traj[:-1:,:,:]

    pred_traj_rel = vlstm_gt(obs_traj, obs_traj_rel)
    pred_traj = relative_to_abs(pred_traj_rel, obs_traj[-1])
    pred_center = pred_traj.detach().numpy()

    wh = np.repeat(np.mean(all_obs[:,:,4:], axis=0, keepdims=True), pred_len, axis=0)
    all_pred[:, :, 0] = all_obs[[-1], :, 0] + np.expand_dims(np.arange(pred_len), axis=1) + 1
    all_pred[:, :, 1] = np.repeat(all_obs[[0],:,1], pred_len, axis=0)
    all_pred[:, :, 2:4] = pred_center
    all_pred[:, :, 4:6] = wh

    return all_pred

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    vlstm = torch.load(os.path.join(args.model_dir,'{}_vlstm_o{}_p{}.pt'.format(
        args.train_dataset, args.obs_len, args.pred_len)))

    H_path = os.path.join(args.homo_dir, args.test_dataset + '.txt')
    H_mat = np.loadtxt(H_path, delimiter=',')

    ## Load original tracking data
    data_dir = '../output'
    data_path = os.path.join(data_dir, '{}_sort_fps{}.txt'.format(
        args.test_dataset, args.fps))
    save_file_path = os.path.join(args.save_dir, '{}_sort_fps{}_predict.txt'.format(
        args.test_dataset, args.fps))

    track_all = np.loadtxt(data_path, delimiter=',')
    track_center_img = track_all[:,2:4] + 0.5 * track_all[:,4:6]
    track_center_wld = convert_pixel_to_world(track_center_img, H_mat)
    track_all[:, 0] /= (30/args.fps)
    track_all[:, 2:4] = track_center_wld # [frame, id, world_x, world_y, w, h]

    all_frame = np.sort(np.unique(track_all[:, 0])).tolist()
    all_peds = np.unique(track_all[:, 1])
    all_observ = []

    for i in range(len(all_peds)):
        traj_i = track_all[track_all[:,1]==all_peds[i], :]

        # Trajectory is not long enough for prediction
        if traj_i.shape[0] < args.obs_len:
            continue
        num_obs = traj_i.shape[0] - args.obs_len + 1
        obs_i = np.zeros((args.obs_len, num_obs, traj_i.shape[1]))

        for j in range(args.obs_len):
            obs_i[j, :, :] = traj_i[j:num_obs+j, :]
        all_observ.append(obs_i)

    all_obs = np.concatenate(all_observ, axis=1)
    vlstm.eval()
    all_pred = bbox_predict(vlstm, all_obs, args)

    # Swapping axes is for saving the trajectory in order
    pred_all = np.reshape(np.swapaxes(all_pred,0,1), (-1, all_pred.shape[2])) # [frame, id, world_x, world_y, w, h]
    pred_center_wld = pred_all[:,2:4]
    pred_center_img = convert_world_to_pixel(pred_center_wld, H_mat)
    pred_all[:,2:4] = pred_center_img - 0.5 * pred_all[:,4:6] # [frame, id, x1, y1, w, h]
    pred_all[:, 0] *= (30/args.fps)

    np.savetxt(save_file_path, pred_all, fmt='%1.3f', delimiter=',')


if __name__ == '__main__':
    main(args)
