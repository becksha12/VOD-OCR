import math
import video_utils
import cv2
import numpy as np
import pytesseract
import time
import string

AD_CIRCLE_COLOR_BOUNDARY_ORANGE = ([50, 115, 195], [125, 155, 255])
AD_CIRCLE_COLOR_BOUNDARY_GRAY = [(40, 40, 40), [55, 55, 55]]
PLAYBAR_COLOR_BOUNDARY_WHITE = ([253, 253, 253], [255, 255, 255])
PLAYBAR_COLOR_BOUNDARY_GRAY = ([75, 75, 75], [77, 77, 77])
DIGIT_DETECTION_CONFIG = '--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789'
TIME_DETECTION_CONFIG = '--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/:'
AD_BREAK_CIRCLE_AREA_RANGE = (20, 35)
PLAYBAR_AREA_TOP_LEFT = (40, 890) # (20, 690)
PLAYBAR_AREA_BOT_RIGHT = (1875, 950) # (1845, 750) #
AD_CIRCLE_TOP_LEFT = (110, 935) # (50, 985)
AD_CIRCLE_BOT_RIGHT = (220, 1040) # (118, 1054)
IMAGE_DISPLAY_DOWNSIZE_RATIO = 0.75


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
    if np.product(window.shape) != 0:
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


def detect_ad_breaks(image, playbar_y_coord):
    ad_break_mask = detect_color_mask(image, AD_CIRCLE_COLOR_BOUNDARY_ORANGE)
    ad_break_mask = cv2.bitwise_not(ad_break_mask)
    cv2.imwrite('ad-breaks-mask.png', ad_break_mask)
    # Find contours
    cnts = cv2.findContours(ad_break_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    ad_coordinates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        ((x, y), r) = cv2.minEnclosingCircle(cnt)
        if AD_BREAK_CIRCLE_AREA_RANGE[0] <= area <= AD_BREAK_CIRCLE_AREA_RANGE[1] and abs(playbar_y_coord - y) <= 6:
            ad_coordinates.append((x, y))
    return ad_coordinates


def get_lowest_horizontal_line(lines, length_threshold):
    lines = lines.squeeze(1).tolist()
    groups = dict()
    for line in lines:
        if line[1] not in groups:
            groups[line[1]] = list()
        groups[line[1]].append(line)

    combined_keys = coalesce(list(groups.keys()))
    combined_groups = {ck: [] for ck in combined_keys}
    for key in groups:
        for cks in combined_keys:
            if key in cks:
                combined_groups[cks].extend(groups[key])

    groups = {int(sum(cks)/len(cks)): combined_groups[cks] for cks in combined_groups}

    continuous_lines = dict()
    for key in groups:
        min_x = min([line[0] for line in groups[key]])
        max_x = max([line[2] for line in groups[key]])
        if max_x - min_x >= length_threshold:
            continuous_lines[key] = (max_x, min_x)

    if len(continuous_lines.keys()) > 0:
        max_y = max(continuous_lines.keys())
        max_x, min_x = continuous_lines[max_y]
        return (min_x, max_y), (max_x, max_y)


def get_playback_time_area(image, playbar_x_min, playbar_x_max, playbar_y):
    area_min_y = playbar_y + 15
    area_max_y = playbar_y + 45
    area_max_x = (playbar_x_min + playbar_x_max) // 4
    area_min_x = playbar_x_min - 25
    return image[area_min_y:area_max_y, area_min_x:area_max_x]


def coalesce(keys, max_difference=3):
    combined_keys = [[keys[0]]]
    for key in keys[1:]:
        to_break = False
        for combined_key_list in combined_keys:
            for inner_key in combined_key_list:
                if abs(inner_key - key) <= max_difference:
                    combined_key_list.append(key)
                    to_break = True
                    break
            if to_break:
                break
        if not to_break:
            combined_keys.append([key])
    return [tuple(ck) for ck in combined_keys]


def get_playbar_coordinates(playbar_mask):
    lines2 = cv2.HoughLinesP(playbar_mask, rho=1, theta=math.pi / 2, threshold=70)
    if lines2 is None:
        return None
    lowest_horizontal_line = get_lowest_horizontal_line(lines2, 900)
    return lowest_horizontal_line


def post_process_time_string(time_string):
    start = 0
    while time_string[start] not in string.digits:
        start += 1
    end = len(time_string) - 1
    while time_string[end] not in string.digits:
        end -= 1
    return time_string[start:end+1]


def parse_playtime(time_string):
    if len(time_string) == 0:
        return
    if time_string is None:
        return
    time_strings = time_string.split('/')
    if len(time_strings) != 2:
        return None
    current_time, total_time = time_string.split('/')
    if len(total_time) < 3:
        return None
    total_time = post_process_time_string(total_time)
    tot_time_splits = total_time.split(':')
    if len(tot_time_splits) < 3:
        return None
    hh, mm, ss, *_ = tot_time_splits
    return int(hh), int(mm), int(ss)



def hhmmss_to_seconds(hh, mm, ss):
    return hh * 3600 + mm * 60 + ss


def seconds_to_hhmmss(seconds):
    hh = int(seconds / 3600)
    remaining = seconds - (hh * 3600)
    mm = int(remaining / 60)
    remaining = remaining - (mm * 60)
    return hh, mm, remaining


def fraction_of_total(point, start, end):
    return (float(point) - start) / (float(end) - start)


def detect_playbar_in_subsample(mask):
    # mask = detect_playbar(subsample, PLAYBAR_COLOR_BOUNDARY_GRAY, PLAYBAR_COLOR_BOUNDARY_WHITE)
    cv2.imwrite('subsam-mask.png', mask)
    lines = cv2.HoughLinesP(mask, rho=1, theta=math.pi / 2, threshold=20)
    if lines is None:
        return None
    return get_lowest_horizontal_line(lines, 900)


def detect_ad_circle_in_subsample(subsample):
    gray = cv2.cvtColor(subsample, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2.0, 60)
    if circles is None:
        return
    return circles


def ad_circle_subsample(frame):
    image = frame
    top_x, top_y = AD_CIRCLE_TOP_LEFT
    bot_x, bot_y = AD_CIRCLE_BOT_RIGHT
    subsample = image[top_y:bot_y, top_x: bot_x]
    cv2.imwrite('ad-circle-subsam.png', subsample)
    circle = detect_ad_circle_in_subsample(subsample)
    if circle is not None:
        coords = circle.squeeze(0).squeeze(0)
        x, y, r = coords
        time = detect_ad_time(subsample, apply_thresholding=True)
        return time
    return None


def playbar_ads_detection_subsample(frame):
    image = frame
    top_x, top_y = PLAYBAR_AREA_TOP_LEFT
    bot_x, bot_y = PLAYBAR_AREA_BOT_RIGHT
    subsample = image[top_y:bot_y, top_x: bot_x]
    # cv2.imwrite('playbar_ads_detection_subsam.png', subsample)
    gray = cv2.cvtColor(subsample, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO)
    line = detect_playbar_in_subsample(mask)
    # print('line:', line)
    if line is not None:
        (x_min, y1), (x_max, y2) = line
        playtime = pytesseract.image_to_string(mask, lang='eng', config=TIME_DETECTION_CONFIG)
        # print('playtime:', playtime)
        hhmmss = parse_playtime(playtime)
        if hhmmss is None:
            return None

        ad_coords = detect_ad_breaks(subsample, y1)
        ad_times = []
        if len(ad_coords) != 0:
            hh, mm, ss = hhmmss
            playtime_in_seconds = hhmmss_to_seconds(hh, mm, ss)
            for i, (x, y) in enumerate(ad_coords):
                add_at_point = max(0, fraction_of_total(x, x_min, x_max))  # TODO: adjust the minimum
                add_at_time = int(add_at_point * playtime_in_seconds)
                h, m, s = seconds_to_hhmmss(add_at_time)
                ad_times.append(f'Add #{i + 1} at: {add_at_time}s ({h:02d}:{m:02d}:{s:02d})')
            return '\n'.join(ad_times)
        else:
            return None
    else:
        return


def main_timer_detection():
    IMAGE_PATH_0 = './data/add.PNG'
    IMAGE_PATH_1 = './data/add-ex-1.PNG'
    IMAGE_PATH_2 = './data/add-ex-2.PNG'
    IMAGE_PATH_3 = './data/add-ex-3.PNG'
    NO_AD_1 = './data/get_add.PNG'
    NO_AD_2 = './data/beast_peacock-ss_1.png'
    FRAME = './problematic-frame.png'
    image = read_image(FRAME)
    ret = detect_ad_circle(image, AD_CIRCLE_COLOR_BOUNDARY_ORANGE, AD_CIRCLE_COLOR_BOUNDARY_GRAY)
    if ret:
        mask, circle = ret
        square = get_timer_window_square_from_circle(circle)
        window = extract_timer_window(image, square)
        print(detect_ad_time(window, True))
    else:
        print('Not an Ad')


def main_playbar_ads_detection():
    IMAGE_1 = './data/get_add.PNG'
    IMAGE_2 = './data/add-break-1.PNG'
    IMAGE_3 = './data/no-ad-breaks.PNG'
    image = read_image(IMAGE_3)
    mask = detect_playbar(image, PLAYBAR_COLOR_BOUNDARY_GRAY, PLAYBAR_COLOR_BOUNDARY_WHITE)
    coordinates = get_playbar_coordinates(mask)
    if coordinates is not None:
        (x_min, y1), (x_max, y2) = coordinates
        timer_area = get_playback_time_area(image, x_min, x_max, y1)
        playtime = pytesseract.image_to_string(timer_area, lang='eng', config=TIME_DETECTION_CONFIG)
        # print(playtime)
        ad_coords = detect_ad_breaks(image, y1)
        if len(ad_coords) == 0:
            ad_coords.sort(key=lambda x: x[0])
            print('ad coords:', ad_coords)
            # print(ad_coords)
            hhmmss = parse_playtime(playtime)
            if hhmmss is not None:
                hh, mm, ss = hhmmss
                playtime_in_seconds = hhmmss_to_seconds(hh, mm, ss)
                for i, (x, y) in enumerate(ad_coords):
                    add_at_point = max(0, fraction_of_total(x, x_min, x_max)) # TODO: adjust the minimum
                    add_at_time = int(add_at_point * playtime_in_seconds)
                    h, m, s = seconds_to_hhmmss(add_at_time)
                    print(f'Add #{i+1} at: {add_at_time}s ({h:02d}:{m:02d}:{s:02d})')
        else:
            print('Playbar not detected')
    else:
        print('Playbar not detected')

    # ((33, 704), (1835, 704))


def main_playbar_ads_detection_subsample():
    IMAGE = './data/add-break-1.PNG'

    image = cv2.imread(IMAGE)
    times = []
    for i in range(100):
        print(i)
        start = time.time()
        top_x, top_y = PLAYBAR_AREA_TOP_LEFT
        bot_x, bot_y = PLAYBAR_AREA_BOT_RIGHT
        subsample = image[top_y:bot_y, top_x: bot_x]
        # print('shape:', subsample.shape)
        # cv2.imwrite('playbar-subsample.png', subsample)
        line = detect_playbar_in_subsample(subsample)
        if line is not None:
            (x_min, y1), (x_max, y2) = line
            # timer_area = get_playback_time_area(subsample, x_min, x_max, y1)
            # cv2.imwrite('timer-area.png', timer_area)
            playtime = pytesseract.image_to_string(subsample, lang='eng', config=TIME_DETECTION_CONFIG)
            hhmmss = parse_playtime(playtime)
            if hhmmss is None:
                print('Playtime not detected')
                return

            ad_coords = detect_ad_breaks(subsample, y1)
            # print(ad_coords)
            if len(ad_coords) != 0:
                hh, mm, ss = hhmmss
                playtime_in_seconds = hhmmss_to_seconds(hh, mm, ss)
                for i, (x, y) in enumerate(ad_coords):
                    add_at_point = max(0, fraction_of_total(x, x_min, x_max))  # TODO: adjust the minimum
                    add_at_time = int(add_at_point * playtime_in_seconds)
                    h, m, s = seconds_to_hhmmss(add_at_time)
                    # print(f'Add #{i + 1} at: {add_at_time}s ({h:02d}:{m:02d}:{s:02d})')
            else:
                print('Ad breaks not detected')
        else:
            print('Playbar not detected')
        end = time.time()
        times.append(end - start)
    print(sum(times) / len(times))


def time_calibration_ad_circle():
    VIDEO_PATH = './data/20221118_223901.mp4'
    times = []
    frame_counter = 0
    for frame in video_utils.get_frame_for_every_second(VIDEO_PATH, 62):
        frame_counter += 1
        print('frame count:', frame_counter)
        start = time.time()
        ret = detect_ad_circle(frame, AD_CIRCLE_COLOR_BOUNDARY_ORANGE, AD_CIRCLE_COLOR_BOUNDARY_GRAY)
        if ret:
            mask, circle = ret
            square = get_timer_window_square_from_circle(circle)
            window = extract_timer_window(frame, square)
            print(detect_ad_time(window, True))
            end = time.time()
            times.append(end - start)
        else:
            print('Not an Ad')
    print('times:', times)
    print('average time:', sum(times) / len(times))


def main_ad_circle_subsample():
    IMAGE = './problematic-frame.png'
    image = cv2.imread(IMAGE)
    VIDEO_PATH = './data/20221118_223901.mp4'
    frame_counter = 0
    for frame in video_utils.get_frame_for_every_second(VIDEO_PATH, 62):
        frame_counter += 1
        image = frame
        top_x, top_y = AD_CIRCLE_TOP_LEFT
        bot_x, bot_y = AD_CIRCLE_BOT_RIGHT
        subsample = image[top_y:bot_y, top_x: bot_x]
        circle = detect_ad_circle_in_subsample(subsample)
        if circle is not None:
            coords = circle.squeeze(0).squeeze(0)
            x, y, r = coords
            time = detect_ad_time(subsample, apply_thresholding=True)
            print(time)
        else:
            print('Ad not detected')


def time_calibration_playbar_det():

    VIDEO_PATH = './data/20221118_225220.mp4'
    times = []
    frame_number = 0
    for image in video_utils.get_frame_for_every_second(VIDEO_PATH, 50):
        # cv2.imwrite(f'playbar_frame_{frame_number}.png', image)
        frame_number += 1
        print('frame number:', frame_number)
        start = time.time()
        mask = detect_playbar(image, PLAYBAR_COLOR_BOUNDARY_GRAY, PLAYBAR_COLOR_BOUNDARY_WHITE)
        coordinates = get_playbar_coordinates(mask)
        if coordinates is not None:
            (x_min, y1), (x_max, y2) = coordinates
            timer_area = get_playback_time_area(image, x_min, x_max, y1)
            playtime = pytesseract.image_to_string(timer_area, lang='eng', config=TIME_DETECTION_CONFIG)
            # print(playtime)
            ad_coords = detect_ad_breaks(image, y1)
            if len(ad_coords) == 0:
                ad_coords.sort(key=lambda x: x[0])
                print('ad coords:', ad_coords)
                # print(ad_coords)
                hhmmss = parse_playtime(playtime)
                if hhmmss is not None:
                    hh, mm, ss = hhmmss
                    playtime_in_seconds = hhmmss_to_seconds(hh, mm, ss)
                    for i, (x, y) in enumerate(ad_coords):
                        add_at_point = max(0, fraction_of_total(x, x_min, x_max))  # TODO: adjust the minimum
                        add_at_time = int(add_at_point * playtime_in_seconds)
                        h, m, s = seconds_to_hhmmss(add_at_time)
                        print(f'Add #{i + 1} at: {add_at_time}s ({h:02d}:{m:02d}:{s:02d})')
            else:
                print('Playbar not detected')
        else:
            print('Playbar not detected')
        end = time.time()
        times.append(end - start)
    avg = sum(times) / len(times)
    print('times:', times)
    print('avg:', avg)


# function for video streaming
def video_stream():
    global count
    frame_id = cap.get(1)
    _, frame = cap.read()
    if frame is not None:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_id % math.floor(frame_rate // 2) == 0:
            cv2.imwrite(f'demo_2_frames/demo_{count:06d}.png', frame)
            ad_timer = ad_circle_subsample(frame)
            if ad_timer is not None:
                # cv2.putText(cv2image, f'Ad timer detected: {ad_timer.strip()}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                time_detected = ad_timer.strip()
                if len(time_detected) > 0:
                    print(f'Ad timer detected: {time_detected}')
            else:
                ad_breaks = playbar_ads_detection_subsample(frame)
                if ad_breaks is not None:
                    # cv2.putText(cv2image, ad_breaks, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    #             cv2.LINE_AA)
                    print('Ad Breaks:')
                    print(ad_breaks)
            count += 1

        cv2image = cv2.resize(cv2image, None, fx=IMAGE_DISPLAY_DOWNSIZE_RATIO, fy=IMAGE_DISPLAY_DOWNSIZE_RATIO, interpolation=cv2.INTER_AREA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(1, video_stream)


if __name__ == '__main__':
    from tkinter import *
    from PIL import ImageTk, Image

    VIDEO_PATH = 'data/rec_14_test.mp4'
    count = 1

    root = Tk()
    # Create a frame
    app = Frame(root, bg="white")
    app.grid()
    # Create a label in the frame
    lmain = Label(app)
    lmain.grid()

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_rate = cap.get(5)

    video_stream()
    root.mainloop()
    # FRAME_PATH = 'demo_2_frames/demo_000125.png'
    # image = cv2.imread(FRAME_PATH)
    # # playbar_ads_detection_subsample(image)
    # print(ad_circle_subsample(image))

