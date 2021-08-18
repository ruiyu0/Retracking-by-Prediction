import os
import cv2
import numpy as np
import argparse
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='../output', type=str)
parser.add_argument('--dataset', default='little3', type=str)
parser.add_argument('--data_dir', default='/home/rzy54/Stanford_Drone/videos/little/video3', type=str)
parser.add_argument('--frame_span', default=7, type=int, help="The whole frame span for compare pairwise frames")
parser.add_argument('--frame_shift', default=5, type=int, help="The frame shift between two frames for comparison")
parser.add_argument('--thresh', default=9, type=int, help="The threshold for binarization")
parser.add_argument('--min_size_for_movement', default=3000, type=int, help="The minimum size of bounding box")
args = parser.parse_args()


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cap = cv2.VideoCapture(os.path.join(args.data_dir, 'video.mov'))
    video_wdth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_hight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_queue = deque()
    frame_num = 0

    print("--------------------- {} - detection start ---------------------".format(args.dataset))

    with open(os.path.join(args.save_dir, "{}_detection.txt".format(args.dataset)),'w') as out_file:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("CAPTURE ERROR / END OF VIDEO")
                break

            # save a greyscale version of the image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Blur it to remove camera noise (reducing false positives)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            frames_queue.append(gray)

            # At the beginning of a video, wait for the length of the queue
            if frame_num <= args.frame_span:
                frame_num += 1
                continue
            frames_queue.popleft()

            # Compare the frames, find the difference
            frame_delta = 0
            for i in range(args.frame_span-args.frame_shift+1):
                frame_delta = np.maximum(frame_delta, cv2.absdiff(frames_queue[i], frames_queue[i+args.frame_shift]))

            frame_thresh = cv2.threshold(frame_delta, args.thresh, 255, cv2.THRESH_BINARY)[1]

            # Fill in holes via dilate(), and find contours of the thesholds
            kernel = np.ones((6, 6), 'uint8')
            frame_dilate = cv2.dilate(frame_thresh, kernel, iterations = 1)
            cnts, _ = cv2.findContours(frame_dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                # If the contour is too small, ignore it, otherwise, there's transient movement
                if cv2.contourArea(c) > args.min_size_for_movement:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    print('%d,-1,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame_num,x,y,w,h),file=out_file)
            frame_num += 1

            if frame_num % 200 == 0:
                print('Processed: {} / {} frames.'.format(frame_num, frame_count))

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main(args)
