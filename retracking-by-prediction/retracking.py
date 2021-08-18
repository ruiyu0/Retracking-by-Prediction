from __future__ import print_function
import torch
import os
import numpy as np
import argparse
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment as linear_assignment
from utils import relative_to_abs, convert_pixel_to_world, convert_world_to_pixel, holt_winters, prune_track, interp_track

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../output', type=str)
    parser.add_argument('--data_dir', default='../output', type=str)
    parser.add_argument('--model_dir', default='saved_models', type=str)
    parser.add_argument('--homo_dir', default='../data/homography', type=str)
    parser.add_argument('--train_dataset', default='trajnet_train', type=str)
    parser.add_argument('--test_dataset', default='little3', type=str)
    parser.add_argument('--obs_len', default=4, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--fps', default=2.5, type=float)
    parser.add_argument('--max_age', default=8, type=int)
    parser.add_argument('--dist_thrld', default=3.0, type=float)
    parser.add_argument('--min_frames', default=2, type=float)
    parser.add_argument('--fill_gap', default=True, type=bool, help='Fill the observation gap when initializing a track')
    parser.add_argument('--min_dist_thrld', default=0.15, type=float, help='If less than it, assign a negative dist (large confidence)') 
    parser.add_argument('--min_hit', default=2, type=float)
    args = parser.parse_args()
    return args


class predict_tracker(object):
    '''
    In this class, observations mean the predicted trajectories, not "observation" for prediction.
    '''
    def __init__(self, track_old, pred_old, pred_model, h_mat, params):
        self.track_old = track_old
        self.pred_old = pred_old
        self.pred_model = pred_model
        self.h_mat = h_mat
        self.params = params
        self.trk_finished = OrderedDict()
        self.trk_active = OrderedDict()
        self.id_count = 0
        self.fake_inf = 100.0

    def update(self, frame):
        self.frame = frame
        max_age = self.params['max_age']
        min_hit = self.params['min_hit']

        # obs_trk may has more IDs than obs_pred
        obs_trk = self.track_old[self.track_old[:,0]==frame, :]
        obs_pred = self.pred_old[:, self.pred_old[0,:,0]==frame+1, :]
        num_obs = obs_pred.shape[1]
        old_ids = obs_pred[0,:,1]

        # If there are no observations
        if num_obs == 0:
            # At the beginning stage, ignore it
            if not bool(self.trk_active):
                return
            # During re-tracking, add age to trackers and check max_age
            else:
                active_ids = list(self.trk_active.keys())
                for trk_id in active_ids:
                    self.trk_active[trk_id]['age'] += 1
                    if self.trk_active[trk_id]['age'] > max_age:
                        self.trk_finished[trk_id] = self.trk_active[trk_id]
                        self.trk_active.pop(trk_id)
        else:
            if not bool(self.trk_active):
                # If active tracker set is empty, initialize it with new observations
                self.initialize_track(obs_trk, obs_pred)
            else:
                matched_obs, matched_trk, unmatched_obs, unmatched_trk = self.hungarian_matching(obs_pred)
                active_ids = list(self.trk_active.keys())

                # For matched tracks, update trackers
                if matched_trk.size > 0:
                    for i, ind in enumerate(matched_trk):
                        m_id = active_ids[ind]
                        self.trk_active[m_id]['age'] = 0
                        self.trk_active[m_id]['hit'] += 1

                        ## update observations
                        self.trk_active[m_id]['obs'] = np.concatenate(
                            (self.trk_active[m_id]['obs'], obs_trk[obs_trk[:,1]==old_ids[matched_obs[i]], :]), axis=0)
                        ## interpolate interrupted points if neccesary                        
                        if self.trk_active[m_id]['obs'][-1,0]-self.trk_active[m_id]['obs'][0,0]+1 > self.trk_active[m_id]['obs'].shape[0]:
                            self.trk_active[m_id]['obs'] = self.interp_traj(self.trk_active[m_id]['obs'])
                        ## update output points
                        if self.trk_active[m_id]['hit'] >= min_hit:
                            self.trk_active[m_id]['pos'] = self.trk_active[m_id]['obs']
                        ## update predictions
                        if self.trk_active[m_id]['obs'].shape[0] >= self.params['obs_len']:
                            smoothed_pos = self.holt_smooth(self.trk_active[m_id]['obs'])
                            self.trk_active[m_id]['pred'] = self.bbox_predict(smoothed_pos)
                            self.trk_active[m_id]['all_pred'].append(self.trk_active[m_id]['pred'])

                # For unmatched tracks, add age and check max_age
                if unmatched_trk.size > 0:
                    for ind in unmatched_trk:
                        m_id = active_ids[ind]
                        self.trk_active[m_id]['age'] += 1
                        self.trk_active[m_id]['hit'] = 0
                        if self.trk_active[m_id]['age'] > max_age:
                            self.trk_finished[m_id] = self.trk_active[m_id]
                            self.trk_active.pop(m_id)

                # For unmatched observation, start a new track
                if unmatched_obs.size > 0:
                    self.initialize_track(obs_trk, obs_pred[:, unmatched_obs, :])

    def hungarian_matching(self, obs_pred):
        dist_thrld = self.params['dist_thrld']
        min_dist_thrld = self.params['min_dist_thrld']
        num_obs = obs_pred.shape[1]
        trk_active = list(self.trk_active.items())
        dist_matrix = self.fake_inf * np.ones((num_obs,len(trk_active)), dtype=np.float32)

        for i in range(num_obs):
            for j, trk in enumerate(trk_active):
                dist_matrix[i, j] = np.amax(self.compute_traj_dist(obs_pred[:,i,:], trk[1]['pred']))

        dist_matrix[dist_matrix > dist_thrld] = self.fake_inf  # Strict matching
        dist_matrix[dist_matrix < min_dist_thrld] = -self.fake_inf # Insure very confident pair to get matched
        obs_ind, trk_ind = linear_assignment(dist_matrix)

        # Remove the assignments larger than threshold
        matched_idx = dist_matrix[obs_ind, trk_ind] <= dist_thrld
        matched_obs = obs_ind[matched_idx]
        matched_trk = trk_ind[matched_idx]
        unmatched_obs = np.delete(np.arange(num_obs), matched_obs)
        unmatched_trk = np.delete(np.arange(len(trk_active)), matched_trk)

        return matched_obs, matched_trk, unmatched_obs, unmatched_trk

    def initialize_track(self, obs_trk, obs_pred):
        num_obs = obs_pred.shape[1]
        old_ids = obs_pred[0,:,1]
        for i in range(num_obs):
            self.id_count += 1
            self.trk_active[self.id_count] = {}
            self.trk_active[self.id_count]['all_pred'] = []
            self.trk_active[self.id_count]['age'] = 0
            self.trk_active[self.id_count]['hit'] = 1

            idx_pos = (self.track_old[:,1]==old_ids[i]) & (self.track_old[:,0]<=self.frame) & (self.track_old[:,0]>self.frame-self.params['obs_len'])
            self.trk_active[self.id_count]['obs'] = self.track_old[idx_pos, :]
            if (self.trk_active[self.id_count]['obs'][-1, 0] - self.trk_active[self.id_count]['obs'][0, 0] + 1) > self.trk_active[self.id_count]['obs'].shape[0]:
                self.trk_active[self.id_count]['obs'] = self.interp_traj(self.trk_active[self.id_count]['obs'])
            # In dxHolt version, 'pos' is the same as 'obs'. Smoothing is only for prediction.
            self.trk_active[self.id_count]['pos'] = self.trk_active[self.id_count]['obs']
            smoothed_pos = self.holt_smooth(self.trk_active[self.id_count]['pos'])
            self.trk_active[self.id_count]['pred'] = self.bbox_predict(smoothed_pos)
            self.trk_active[self.id_count]['all_pred'].append(self.trk_active[self.id_count]['pred'])

    def compute_traj_dist(self, obs_pred, trk_pred):
        frames, o_ind, t_ind = np.intersect1d(obs_pred[:,0], trk_pred[:,0], return_indices=True)
        if len(frames)==0:
            return self.fake_inf
        else:
            obs_overlap = obs_pred[o_ind, :]
            trk_overlap = trk_pred[t_ind, :]
            maha_dist = self.compute_Mahalanobis_dist(obs_overlap, trk_overlap)

        return maha_dist
        
    def compute_Mahalanobis_dist(self, obs, trk):
        obs_xy_img = obs[:, 2:4] + 0.5 * obs[:, 4:6]
        trk_xy_img = trk[:, 2:4] + 0.5 * trk[:, 4:6]
        obs_xy_wld = convert_pixel_to_world(obs_xy_img, self.h_mat)
        trk_xy_wld = convert_pixel_to_world(trk_xy_img, self.h_mat)
        obs_omg = obs[:, -1]
        trk_omg = trk[:, -1]

        dist = np.sqrt(((obs_xy_wld[:,0]-trk_xy_wld[:,0])**2 + (obs_xy_wld[:,1]-trk_xy_wld[:,1])**2) / (obs_omg + trk_omg))

        return dist

    def interp_traj(self, positions):
        frames = positions[:,0]
        new_len = int(frames[-1] - frames[0] + 1)
        new_frames = np.arange(frames[0], frames[-1]+1)
        new_positions = np.zeros((new_len, 7))
        new_positions[:,0] = new_frames # frame
        new_positions[:,1] = positions[0,1] # id
        new_positions[:,6] = positions[0,6]

        new_positions[:,2] = np.interp(new_frames, frames, positions[:,2])
        new_positions[:,3] = np.interp(new_frames, frames, positions[:,3])
        new_positions[:,4] = np.interp(new_frames, frames, positions[:,4])
        new_positions[:,5] = np.interp(new_frames, frames, positions[:,5])

        return new_positions

    def holt_smooth(self, positions, alpha=0.5, beta=0.5):
        new_positions = np.copy(positions)

        centers = positions[:,2:4] + 0.5 * positions[:,4:6]

        delta_x = centers[1:,0] - centers[:-1,0]
        delta_y = centers[1:,1] - centers[:-1,1]

        new_delta_x = holt_winters(delta_x, alpha, beta)
        new_delta_y = holt_winters(delta_y, alpha, beta)

        ## Reversely got the smoothed past positions
        new_centers = np.repeat(centers[[-1],:], centers.shape[0], axis=0)
        new_centers[:-1,0] -= np.cumsum(new_delta_x[::-1])[::-1]
        new_centers[:-1,1] -= np.cumsum(new_delta_y[::-1])[::-1]

        new_positions[:,2:4] = new_centers - 0.5 * new_positions[:,4:6]

        return new_positions

    def bbox_predict(self, observations):
        self.pred_model.eval()
        obs_len = self.params['obs_len']
        pred_len = self.params['pred_len']
        seq_len = obs_len + pred_len

        predictions = np.zeros((pred_len, 7))
        predictions[:,0] = observations[-1,0] + np.arange(pred_len) + 1 # frame
        predictions[:,1] = observations[0,1] # id

        obs_center_img = observations[-obs_len:,2:4] + 0.5 * observations[-obs_len:,4:6]
        obs_center_wld = convert_pixel_to_world(obs_center_img, self.h_mat)
        obs_center = np.expand_dims(obs_center_wld, axis=1)

        obs_traj = torch.from_numpy(obs_center).type(torch.float)
        obs_traj_rel = torch.zeros(obs_traj.shape, dtype=torch.float)
        obs_traj_rel[1:,:,:] = obs_traj[1:,:,:] - obs_traj[:-1:,:,:]

        pred_traj_rel = self.pred_model(obs_traj, obs_traj_rel)
        pred_traj = relative_to_abs(pred_traj_rel, obs_traj[-1])
        pred_center = pred_traj.detach().numpy()

        pred_center_wld = np.squeeze(pred_center)
        pred_center_img = convert_world_to_pixel(pred_center_wld, self.h_mat)

        predictions[:,4] = np.mean(observations[:,4]) # w
        predictions[:,5] = np.mean(observations[:,5]) # h
        predictions[:,2:4] = pred_center_img - 0.5 * predictions[:,4:6] # x1, y1
        predictions[:,6] = np.arange(1, pred_len+1)

        return predictions

    def get_active_tracks(self):
        obs_len = self.params['obs_len']
        tracks = []
        predicts = []
        for trk_id, trk in self.trk_active.items():
            trk['pos'][:, 1] = trk_id
            if self.params['fill_gap']:
                tracks.append(trk['pos'])
            else:
                tracks.append(trk['pos'][obs_len-1:,])
            pred = np.concatenate(trk['all_pred'], axis=0)
            pred[:, 1] = trk_id
            predicts.append(pred)
        active_tracks = np.concatenate(tracks, axis=0)
        active_predicts = np.concatenate(predicts, axis=0)

        return active_tracks, active_predicts

    def get_finished_tracks(self):
        obs_len = self.params['obs_len']
        tracks = []
        predicts = []
        for trk_id, trk in self.trk_finished.items():
            trk['pos'][:, 1] = trk_id
            if self.params['fill_gap']:
                tracks.append(trk['pos'])
            else:
                tracks.append(trk['pos'][obs_len-1:,])
            pred = np.concatenate(trk['all_pred'], axis=0)
            pred[:, 1] = trk_id
            predicts.append(pred)
        finished_tracks = np.concatenate(tracks, axis=0)
        finished_predicts = np.concatenate(predicts, axis=0)

        return finished_tracks, finished_predicts


def main():
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
 
    # Load prediction model Vanilla LSTM
    vlstm = torch.load(os.path.join(args.model_dir,'{}_vlstm_o{}_p{}.pt'.format(
        args.train_dataset, args.obs_len, args.pred_len)))

    H_path = os.path.join(args.homo_dir, args.test_dataset + '.txt')
    H_mat = np.loadtxt(H_path, delimiter=',')

    # Load SORT tracking data
    data_path = os.path.join(args.data_dir, '{}_sort_fps{}.txt'.format(args.test_dataset, args.fps))
    sort_all = np.loadtxt(data_path, delimiter=',')

    # Load offline prediction data
    data_path = os.path.join(args.data_dir, '{}_sort_fps{}_predict.txt'.format(args.test_dataset, args.fps))
    pred_all_2d = np.loadtxt(data_path, delimiter=',')
    pred_all_3d = np.swapaxes(np.reshape(pred_all_2d, (-1, args.pred_len, pred_all_2d.shape[1])), 0,1)

    # Nomalize the frame number (for re-tracking)
    sort_all[:, 0] /= (30/args.fps)
    pred_all_3d[:, :, 0] /= (30/args.fps)
    all_frame = np.sort(np.unique(sort_all[:, 0])).tolist()

    # Add an additional dimension to represent the predict step (0 means no prediction)
    track_all = np.hstack((sort_all, np.zeros((sort_all.shape[0],1))))
    pred_all = np.zeros((args.pred_len, pred_all_3d.shape[1], pred_all_3d.shape[2]+1))
    pred_all[:, :, :-1] = pred_all_3d
    pred_all[:, :, -1] = np.expand_dims(np.arange(1, args.pred_len+1), axis=1)

    print("--------------------- {} - retracking start ---------------------".format(args.test_dataset))

    params = vars(args)
    re_tracker = predict_tracker(track_old=track_all, pred_old=pred_all, pred_model=vlstm, h_mat=H_mat, params=params)

    # For each frame, do prediction and re-tracking
    for i in range(len(all_frame)):
        frame = all_frame[i]
        re_tracker.update(frame)

        if (i+1) % 100 == 0:
            print('Finished {}/{}'.format(i+1, len(all_frame)))

    # Get the re-tracked results
    retrack_finished, predict_finished = re_tracker.get_finished_tracks()
    retrack_active, predict_active  = re_tracker.get_active_tracks()

    retrack_all = np.vstack((retrack_finished, retrack_active))
    retrack_all = retrack_all[:,:6]
    retrack_all = interp_track(retrack_all)
    if args.fill_gap:
        min_frames = args.min_frames + args.obs_len
    retrack_all = prune_track(retrack_all, min_frames)

    predict_all = np.vstack((predict_finished, predict_active))
    predict_all = predict_all[:,:6]

    # Recover the frame number
    retrack_all[:, 0] *= (30/args.fps)
    predict_all[:, 0] *= (30/args.fps)

    save_file_path = os.path.join(args.save_dir, '{}_retrack.txt'.format(args.test_dataset))
    np.savetxt(save_file_path, retrack_all, fmt='%1.3f', delimiter=',')

    save_file_path = os.path.join(args.save_dir, '{}_retrack_predict.txt'.format(args.test_dataset))
    np.savetxt(save_file_path, predict_all, fmt='%1.3f', delimiter=',')

    print('Finished re-tracking'.format(args.test_dataset))


if __name__ == '__main__':
    main()
