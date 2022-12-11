import numpy as np
import imutils
import cv2
import config
import video_utils


def read_template(template_path, reduce_size=False):
    # load the image image, convert it to grayscale, and detect edges
    template = cv2.imread(template_path)
    if reduce_size:
        template = cv2.resize(template, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    template_h, template_w = template.shape[:2]
    return template, template_h, template_w


def read_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def norm_matrix(matrix):
    normed = (matrix - np.mean(matrix)) / np.std(matrix)
    return normed


def match(image_gray, template, template_width, template_height):
    found = None
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(image_gray, width=int(image_gray.shape[1] * scale))
        r = image_gray.shape[1] / float(resized.shape[1])
        if resized.shape[0] < template_height or resized.shape[1] < template_width:
            break
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        normed_result = norm_matrix(result)
        norm_max = normed_result.max()
        norm_coords = np.unravel_index(normed_result.argmax(), normed_result.shape)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r, norm_max, norm_coords)
    if found[-2] > config.KEYBOARD_TEMPLATE_CORR_THRESHOLD:
        (max_val, maxLoc, r, norm_max, norm_coords) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + template_width) * r), int((maxLoc[1] + template_height) * r))
        return (startX, startY), (endX, endY), max_val, norm_max, norm_coords


def display_bounding_box(image, start_x, start_y, end_x, end_y, score=None, save_path='template_matching_output.png'):
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 4)
    if score is not None:
        cv2.putText(image, str(round(score, 3)), (start_x + 10, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(save_path, image)


def main():
    # FILE_PATH = './data/beast_peacock-ss_1.png'
    # FILE_PATH = './data/peacock-ss_2.png'
    # TEMPLATE_PATH = './data/template-2-full-kbd.png'
    #
    # output_path = './outputs/' + FILE_PATH.split('/')[-1]
    # image, gray = read_image(FILE_PATH)
    # template, template_h, template_w = read_template(TEMPLATE_PATH)
    # coordinates = match(gray, template, template_w, template_h)
    # if coordinates is not None:
    #     print('Match found')
    #     (sx, sy), (ex, ey), score, norm_max, norm_coords = coordinates
    #     display_bounding_box(image, sx, sy, ex, ey, score=score, save_path=output_path)
    #     print((sx, sy), (ex, ey), score, norm_max, norm_coords)
    #
    VIDEO_PATH = './data/rec_keybrd_01.mp4'
    OUTPUT_DIR = './framewise_preds'
    TEMPLATE_PATH = './data/template-2-full-kbd.png'
    counter = 0
    for image in video_utils.get_frame_for_every_second(VIDEO_PATH):
        counter += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template, template_h, template_w = read_template(TEMPLATE_PATH)
        coordinates = match(gray, template, template_w, template_h)
        output_path = f'{OUTPUT_DIR}/frame_{counter:04d}.png'
        if coordinates is not None:
            print(counter, 'Match found')
            (sx, sy), (ex, ey), score, norm_max, norm_coords = coordinates
            display_bounding_box(image, sx, sy, ex, ey, score=norm_max, save_path=output_path)
        else:
            cv2.imwrite(output_path, image)


if __name__ == '__main__':
    main()
