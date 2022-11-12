import cv2
import math


def get_frame_for_every_second(video_path):
    vid_ob = cv2.VideoCapture(video_path)
    frame_rate = vid_ob.get(5)
    while vid_ob.isOpened():
        frame_id = vid_ob.get(1)
        ret, frame = vid_ob.read()
        if ret:
          if frame_id % math.floor(frame_rate) == 0:
              yield frame
    vid_ob.release()
