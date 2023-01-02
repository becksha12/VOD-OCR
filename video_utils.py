import cv2
import math


def get_N_frames_for_every_second(video_path, N=1, max_frames=None):
    vid_ob = cv2.VideoCapture(video_path)
    frame_rate = vid_ob.get(5)
    if max_frames is None:
        max_frames = math.inf
    current_frame = 0
    while vid_ob.isOpened() and current_frame < max_frames:
        frame_id = vid_ob.get(1)
        ret, frame = vid_ob.read()
        if ret:
          if frame_id % math.floor(frame_rate // N) == 0:
              current_frame += 1
              yield frame
        else:
            break
    vid_ob.release()


