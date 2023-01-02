import cv2
import video_utils
import numpy as np


def get_frame_entropy(frame):
    marg = np.histogramdd(np.ravel(frame), bins=256)[0] / frame.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    return entropy


def laplacian_entropy(frame):
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    return get_frame_entropy(laplacian)


if __name__ == '__main__':
    VIDEO_PATH = 'data/VHO11_SG2_STB04_2022-12-29T14_47_39.150Z.mp4'
    counter = 0
    N = 10
    seconds = 0
    for frame in video_utils.get_N_frames_for_every_second(VIDEO_PATH, N):
        counter += 1
        seconds += 1 / N
        file_name = f'corrupt_frames_det/frame_{counter:03d}.png'
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        entropy = get_frame_entropy(gray)
        lap_entropy = laplacian_entropy(gray)
        if entropy == 0:
            print(f'Corrupt frame at: {seconds}s', file_name, 'Entropy:', entropy)
            cv2.imwrite(file_name, frame)
        else:
            ratio = lap_entropy / entropy
            if ratio < 0.38:
                print(f'Corrupt frame at: {seconds}s', file_name, 'Entropy Ratio:', ratio)
                cv2.imwrite(file_name, frame)
