"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    --------------------------------------------------------------------
    Rui Yu (rzy54@psu.edu)
    Change IoU to L2 distance for association.
"""

from __future__ import print_function

import os.path
import numpy as np
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment as linear_assignment
import time
import argparse
from filterpy.kalman import KalmanFilter


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--save_dir', default='../output', type=str)
    parser.add_argument('--dataset', default='little3', type=str)
    parser.add_argument('--maxAge', default=10, type=int)
    parser.add_argument('--minHits', default=5, type=int)
    parser.add_argument('--distThre', default=60, type=int)
    args = parser.parse_args()
    return args

def l2_dist(bb_test, bb_gt):
    """
    Computes center L2 distance between two bboxes in the form [x1,y1,x2,y2]
    """
    test_x = 0.5 * (bb_test[0] + bb_test[2])
    test_y = 0.5 * (bb_test[1] + bb_test[3])
    gt_x = 0.5 * (bb_gt[0] + bb_gt[2])
    gt_y = 0.5 * (bb_gt[1] + bb_gt[3])
    dist = np.sqrt((test_x-gt_x)**2 + (test_y-gt_y)**2)
    return dist

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h    #scale is just area
    r = w/float(h)
    return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

def associate_detections_to_trackers_dist(detections,trackers,dist_threshold=60):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    dist_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            dist_matrix[d,t] = l2_dist(det,trk)
    matched_indices = linear_assignment(dist_matrix)

    matched_indices = np.vstack(matched_indices).T # for new linear_assignment

    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t,trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with large distance
    matches = []
    for m in matched_indices:
        if(dist_matrix[m[0],m[1]]>dist_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0], \
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.001
        self.kf.Q[4:,4:] *= 0.001

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class Sort(object):
    def __init__(self,max_age=1,min_hits=3,dist_thres=40):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.dist_threshold = dist_thres

    def update(self,dets):
        """
        Params:
            dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5))
        to_del = []
        ret = []
        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers_dist(dets,trks,dist_threshold=self.dist_threshold)

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1]==t)[0],0]
                trk.update(dets[d,:][0])

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i,:]) 
                self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
                d = trk.get_state()[0]
                if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                    ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
                i -= 1
                #remove dead tracklet
                if(trk.time_since_update > self.max_age):
                    self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    total_time = 0.0
    total_frames = 0

    mot_tracker = Sort(max_age=args.maxAge, min_hits=args.minHits, dist_thres=args.distThre)
    seq_dets = np.loadtxt('../output/{}_detection.txt'.format(args.dataset),delimiter=',')

    start_frame = seq_dets[:,0].min()   
    seq_dets[:,0] = seq_dets[:,0] - start_frame + 1

    print("--------------------- {} - tracking start ---------------------".format(args.dataset))

    with open(os.path.join(args.save_dir, '{}_sort.txt'.format(args.dataset)),'w') as out_file:
        for frame in range(int(seq_dets[:,0].max())):
            frame += 1 # detection and frame numbers begin at 1
            dets = seq_dets[seq_dets[:,0]==frame,2:7]
            dets[:,2:4] += dets[:,0:2] # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            total_frames += 1

            start_time = time.time()
            trackers = mot_tracker.update(dets)
            cycle_time = time.time() - start_time
            total_time += cycle_time
    
            true_frame = frame + start_frame - 1

            for d in trackers:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(true_frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)

    print("Finished SORT tracking with %.3f sec for %d frames"%(total_time, total_frames))
