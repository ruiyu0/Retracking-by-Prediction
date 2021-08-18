import numpy as np
import torch


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

def rand_rotate(batch):
    angle = np.random.randn(1) * 2 * np.pi

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel) = batch
    obs_traj_np = obs_traj.numpy()
    pred_traj_gt_np = pred_traj_gt.detach().numpy()
    obs_traj_rel_np = obs_traj_rel.detach().numpy()
    pred_traj_gt_rel_np = pred_traj_gt_rel.detach().numpy()

    obs_traj = torch.from_numpy(rotate(obs_traj_np[:,:,0], obs_traj_np[:,:,1], angle)).type(torch.float)
    pred_traj_gt = torch.from_numpy(rotate(pred_traj_gt_np[:,:,0], pred_traj_gt_np[:,:,1], angle)).type(torch.float)
    obs_traj_rel = torch.from_numpy(rotate(obs_traj_rel_np[:,:,0], obs_traj_rel_np[:,:,1], angle)).type(torch.float)
    pred_traj_gt_rel = torch.from_numpy(rotate(pred_traj_gt_rel_np[:,:,0], pred_traj_gt_rel_np[:,:,1], angle)).type(torch.float)

    return (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel)

def rotate(x, y, angle):
    dim = len(x.shape)
    (rho, phi) =  cart2pol(x, y)
    x_new, y_new = pol2cart(rho, phi+angle)
    return np.stack((x_new, y_new), axis=dim)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def l2_loss(pred_traj, pred_traj_gt, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def convert_pixel_to_world(pixel_coord, H):
    # pixel_coord - Image coordinates, size (N x 2)
    # H - Homography matrix, size (3 x 3)
    # Return: World position, size (N x 2)

    num = pixel_coord.shape[0]
    pc_homo = np.concatenate((np.transpose(pixel_coord), np.ones((1, num))), axis=0)

    worldP = np.matmul(H, pc_homo)

    world_coord = worldP[:2, :] / worldP[2, :]
    world_coord = np.transpose(world_coord)

    return world_coord

def convert_world_to_pixel(world_coord, H):
    """
    Inputs:
      world_coord - World coordinates, size (N x 2)
      H - Homography matrix, size (3 x 3)
    Return: pixel_coord - Pixel position, size (N x 2)
    """
    num = world_coord.shape[0]
    wc_homo = np.concatenate((np.transpose(world_coord), np.ones((1, num))), axis=0)

    H_inv = np.linalg.inv(H)
    imageP = np.matmul(H_inv, wc_homo)

    pixel_coord = imageP[:2, :] / imageP[2, :]
    pixel_coord = np.transpose(pixel_coord)

    return pixel_coord

def holt_winters(series, alpha, beta, simple=False):
    '''
    Holt-Winters double exponential smoothing
    Assume the time steps are regular (with same interval)
    If simple=True, it becomes basic (simple) exponential smoothing (Holt linear)
    '''
    level = series[0]
    # trend = series[1] - series[0]
    trend = 0 # This initialization is used for delta_x smoothing
    if simple==True:
        trend = 0
        beta = 0
    results = np.zeros(series.size)
    results[0] = level
    for i in range(1, series.size):
        previous_level = level
        previous_trend = trend
        level = alpha * series[i] + (1 - alpha) * (previous_level + previous_trend)
        trend = beta * (level - previous_level) + (1 - beta) * previous_trend
        results[i] = level
    return results

def prune_track(track_all, min_frames):
    uniq_ids = np.unique(track_all[:, 1])
    num_id = len(uniq_ids)
    track_new = []
    for i in range(num_id):
        idx = track_all[:,1] == uniq_ids[i]
        trk_i = track_all[idx, :]
        trk_i = trk_i[trk_i[:,0].argsort()] # Sort by first column (frame) in ascending order
        if trk_i.shape[0] < min_frames: # ignore the id with few frames
            continue
        track_new.append(trk_i)
    track_prune = np.concatenate(track_new, axis=0)

    return track_prune

def interp_track(track_all):
    uniq_ids = np.unique(track_all[:, 1])
    num_id = len(uniq_ids)
    track_new = []
    for i in range(num_id):
        idx = track_all[:,1] == uniq_ids[i]
        trk_i = track_all[idx, :]
        trk_i = trk_i[trk_i[:,0].argsort()] # Sort by first column (frame) in ascending order
        if trk_i.shape[0] == 1: # ignore the id with only one frame
            continue
        frames = np.arange(trk_i[0,0], trk_i[-1,0]+1, 1) # continous frames to be interpolated
        new_trk_i = np.zeros((len(frames),6))
        new_trk_i[:,0] = frames
        new_trk_i[:,1] = uniq_ids[i]
        new_trk_i[:,2] = np.interp(frames, trk_i[:,0], trk_i[:,2])
        new_trk_i[:,3] = np.interp(frames, trk_i[:,0], trk_i[:,3])
        new_trk_i[:,4] = np.interp(frames, trk_i[:,0], trk_i[:,4])
        new_trk_i[:,5] = np.interp(frames, trk_i[:,0], trk_i[:,5])
        track_new.append(new_trk_i)
    track_interp = np.concatenate(track_new, axis=0)

    return track_interp