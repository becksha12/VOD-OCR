import math

import cv2
import numpy as np
import pytesseract

AD_CIRCLE_COLOR_BOUNDARY_ORANGE = ([66, 123, 205], [77, 133, 252])
AD_CIRCLE_COLOR_BOUNDARY_GRAY = [(50, 50, 50), [52, 52, 52]]
PLAYBAR_COLOR_BOUNDARY_WHITE = ([253, 253, 253], [255, 255, 255])
PLAYBAR_COLOR_BOUNDARY_GRAY = ([75, 75, 75], [77, 77, 77])
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


def detect_playbar(image, color_boundary_gray, color_boundary_white):
    gray_mask = detect_color_mask(image, color_boundary_gray)
    white_mask = detect_color_mask(image, color_boundary_white)
    mask = cv2.bitwise_xor(gray_mask, white_mask)
    return mask


def main_timer_detection():
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


def get_lowest_horizontal_line(lines):
    lines = lines.squeeze(1).tolist()
    lines.sort(key=lambda x: x[1], reverse=True)
    max_y_coordinate = lines[0][1]
    lowest_lines = list(filter(lambda l: l[1] == max_y_coordinate, lines))
    min_x = min([l[0] for l in lowest_lines])
    max_x = max([l[2] for l in lowest_lines])
    return (min_x, max_y_coordinate), (max_x, max_y_coordinate)


def get_playbar_coordinates(playbar_mask):
    edges = cv2.Canny(playbar_mask, 80, 120)
    # lines1 = cv2.HoughLinesP(playbar_mask, rho=1, theta=math.pi / 2, threshold=200)
    lines2 = cv2.HoughLinesP(edges, rho=1, theta=math.pi / 2, threshold=70, minLineLength=1000)
    lowest_horizontal_line = get_lowest_horizontal_line(lines2)
    return lowest_horizontal_line


def main_playbar_ads_detection():
    IMAGE_1 = './data/get_add.PNG'
    IMAGE_2 = './data/add-break-1.PNG'
    image = read_image(IMAGE_1)
    mask = detect_playbar(image, PLAYBAR_COLOR_BOUNDARY_GRAY, PLAYBAR_COLOR_BOUNDARY_WHITE)
    cv2.imwrite('playbar_mask.png', mask)
    print(get_playbar_coordinates(mask))
    # ((33, 704), (1835, 704))


if __name__ == '__main__':
    main_playbar_ads_detection()
    # Test pick circles function
    # print(pick_bottom_left_circle(np.array([[[25, 35, 12.5]]])) == [25, 35, 12.5])
    # print(pick_bottom_left_circle(np.array([[[25, 35, 12.5], [25, 100, 12.5], [35, 120, 12.5]]])) == [25, 100, 12.5])
    # print(pick_bottom_left_circle(np.array([[[0, 50, 12.0], [100, 100, 13.6]]])) == [0, 50, 12.0])
