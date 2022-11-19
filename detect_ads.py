import math

import cv2
import numpy as np
import pytesseract

AD_CIRCLE_COLOR_BOUNDARY_ORANGE = ([66, 123, 205], [77, 133, 252])
AD_CIRCLE_COLOR_BOUNDARY_GRAY = [(50, 50, 50), [52, 52, 52]]
DIGIT_DETECTION_CONFIG = '--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789'

def read_image(image_path):
    return cv2.imread(image_path)


def detect_color_mask(image, color_boundary):
    lower, upper = color_boundary
    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv2.inRange(image, lower, upper)
    return mask


def pick_bottom_left_circle(detected_circles):
    circles_list = detected_circles.squeeze(0).tolist()
    # Convert x and y co-ordinates into integers
    circles_list = [[int(x), int(y), round(z, 3)] for x, y, z in circles_list]
    # out of the circles detected pick the one with min(x) and max(y) so that we are looking at bottom left corner
    circles_list.sort(key=lambda circ: circ[0], reverse=False)
    # Keep circles with minimum x
    min_x = circles_list[0][0]
    min_x_circles = list(filter(lambda circ: circ[0] == min_x, circles_list))
    # Sort to get the maximum y
    min_x_circles.sort(key=lambda circ: circ[1], reverse=True)
    return min_x_circles[0]


def detect_ad_circle(image, color_boundary_orange, color_boundary_gray):
    orange_mask = detect_color_mask(image, color_boundary_orange)
    gray_mask = detect_color_mask(image, color_boundary_gray)
    mask = cv2.bitwise_xor(orange_mask, gray_mask)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 2.0, 60)
    if circles is None:
        return

    bottom_left_circle = pick_bottom_left_circle(circles)

    return mask, bottom_left_circle


def get_timer_window_square_from_circle(circle):
    x, y, r = circle
    half_diagonal = math.sqrt(2) * r
    top_left = (int(x - half_diagonal), int(y - half_diagonal))
    bottom_right = (int(x + half_diagonal), int(y + half_diagonal))
    return top_left, bottom_right


def extract_timer_window(image, window_coordinates):
    top_left, bottom_right = window_coordinates
    top_left_x, top_left_y = top_left
    bottom_right_x, bottom_right_y = bottom_right

    return image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]


def detect_ad_time(window, apply_thresholding=True):
    if apply_thresholding:
        gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        _, window = cv2.threshold(gray, 200, 255, cv2.THRESH_TOZERO)
    cv2.imwrite('window_thresholded.png', window)
    return pytesseract.image_to_string(window, lang='eng', config=DIGIT_DETECTION_CONFIG)


def main():
    IMAGE_PATH_0 = './data/add.PNG'
    IMAGE_PATH_1 = './data/add-ex-1.PNG'
    IMAGE_PATH_2 = './data/add-ex-2.PNG'
    IMAGE_PATH_3 = './data/add-ex-3.PNG'
    NO_AD_1 = './data/get_add.PNG'
    NO_AD_2 = './data/beast_peacock-ss_1.png'
    image = read_image(IMAGE_PATH_3)
    ret = detect_ad_circle(image, AD_CIRCLE_COLOR_BOUNDARY_ORANGE, AD_CIRCLE_COLOR_BOUNDARY_GRAY)
    if ret:
        mask, circle = ret
        square = get_timer_window_square_from_circle(circle)
        window = extract_timer_window(image, square)
        print(detect_ad_time(window, True))
    else:
        print('Not an Ad')


if __name__ == '__main__':
    main()
    # Test pick circles function
    # print(pick_bottom_left_circle(np.array([[[25, 35, 12.5]]])) == [25, 35, 12.5])
    # print(pick_bottom_left_circle(np.array([[[25, 35, 12.5], [25, 100, 12.5], [35, 120, 12.5]]])) == [25, 100, 12.5])
    # print(pick_bottom_left_circle(np.array([[[0, 50, 12.0], [100, 100, 13.6]]])) == [0, 50, 12.0])
