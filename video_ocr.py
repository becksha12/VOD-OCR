import cv2
import pytesseract
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


def get_text(image):
    return pytesseract.image_to_string(image)


def main():
    FILE_PATH = './data/iridescent-linkin-park.mp4'
    frame_counter = 0

    for image in get_frame_for_every_second(FILE_PATH):
        text = get_text(image)
        print('Frame #', frame_counter)
        print(text)
        frame_counter += 1


if __name__ == '__main__':
    main()
