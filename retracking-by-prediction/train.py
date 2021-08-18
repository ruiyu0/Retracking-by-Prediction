import torch
import torch.optim as optim
import numpy as np
import argparse
import os
from loader import data_loader
from models import VanillaLSTMNet
from utils import relative_to_abs, rand_rotate, l2_loss

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=16, help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--save_dir', default='./saved_models', type=str)
# dataset options
parser.add_argument('--train_dataset', default='trajnet_train', type=str)
parser.add_argument('--test_dataset', default='trajnet_test', type=str)
# parser.add_argument('--train_type', default='gt', type=str)
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=4, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=2, type=int)

args = parser.parse_args()


def test(args, vanilla_lstm_net, pred_len):
    test_data_dir = './datasets/test/{}'.format(args.test_dataset)
    dataset, dataloader = data_loader(args, test_data_dir)
    test_loss = []

    for i, batch in enumerate(dataloader):
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel) = batch
        out = vanilla_lstm_net(obs_traj, obs_traj_rel)
        pred_traj_fake_rel = out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        cur_test_loss = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, mode='sum')
        test_loss.append(cur_test_loss.item())        

        error1 = pred_traj_fake.detach().numpy() - pred_traj_gt.detach().numpy()
        if i==0:
            total_error1 = error1
        else:
            total_error1 = np.concatenate((total_error1, error1), axis=1)

    avg_testD_error = np.mean(np.sqrt(np.sum(np.square(total_error1),axis=2)))
    avg_testfinalD_error = np.mean(np.sqrt(np.sum(np.square(total_error1[-1,:,:]),axis=1)))
    avg_testloss = sum(test_loss)/len(test_loss)

    return avg_testloss, avg_testD_error, avg_testfinalD_error

def main(args):
   
    data_dir = './datasets/train/{}'.format(args.train_dataset)

    num_epoch = args.num_epochs
    obs_len = args.obs_len
    pred_len = args.pred_len
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    
    dataset, dataloader = data_loader(args, data_dir)

    vanilla_lstm_net = VanillaLSTMNet(obs_len=args.obs_len, pred_len=args.pred_len)
    optimizer = optim.Adam(vanilla_lstm_net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)

    for i in range(num_epoch):

        for t, batch in enumerate(dataloader):

            optimizer.zero_grad()

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel) = batch
            seq, peds, coords = obs_traj.shape
            out = vanilla_lstm_net(obs_traj, obs_traj_rel)
            pred_traj_fake_rel = out
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
            cur_train_loss1 = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, mode='sum')

            # data augmentation (random rotation)
            (obs_traj_rot, pred_traj_gt_rot, obs_traj_rel_rot, pred_traj_gt_rel_rot) = rand_rotate(batch)
            out_rot = vanilla_lstm_net(obs_traj_rot, obs_traj_rel_rot)
            pred_traj_fake_rel_rot = out_rot
            pred_traj_fake_rot = relative_to_abs(pred_traj_fake_rel_rot, obs_traj_rot[-1])
            cur_train_loss2 = l2_loss(pred_traj_fake_rel_rot, pred_traj_gt_rel_rot, mode='sum')
            cur_train_loss = cur_train_loss1 + cur_train_loss2

            if (t+1) % args.print_every == 0:
                print('[Epoch {:3d}/{}]{:3d}, train_loss = {:.3f}'.format(i+1, num_epoch, t+1, cur_train_loss.item()))
            cur_train_loss.backward()
            optimizer.step()
      
        avgTestLoss, avgD_test, finalD_test = test(args, vanilla_lstm_net, pred_len)
        print('[Epoch {:3d}/{}] test_loss = {:5.3f}, test_ADE = {:5.3f}, test_FDE = {:5.3f}'.format(
            i+1, num_epoch, avgTestLoss, avgD_test, finalD_test))

        scheduler.step()

    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
    save_path = os.path.join(args.save_dir, '{}_vlstm_o{}_p{}.pt'.format(args.train_dataset, obs_len, pred_len))
    torch.save(vanilla_lstm_net, save_path)
    print("Finished training vanilla LSTM model")

if __name__ == '__main__':
    main(args)
