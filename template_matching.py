import numpy as np
import argparse
import imutils
import cv2


def read_template(template_path):
    # load the image image, convert it to grayscale, and detect edges
    template = cv2.imread(template_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    template_h, template_w = template.shape[:2]
    return template, template_h, template_w


def read_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


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
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + template_width) * r), int((maxLoc[1] + template_height) * r))
    return (startX, startY), (endX, endY)


def display_bounding_box(image, start_x, start_y, end_x, end_y, save_path='template_matching_output.png'):
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 4)
    cv2.imwrite(save_path, image)


def main():
    FILE_PATH = './data/beast_peacock-ss_1.png'
    TEMPLATE_PATH = './data/template-1.png'

    image, gray = read_image(FILE_PATH)
    template, template_h, template_w = read_template(TEMPLATE_PATH)
    coordinates = match(gray, template, template_w, template_h)
    if coordinates is not None:
        print('Match found')
        (sx, sy), (ex, ey) = coordinates
        display_bounding_box(image, sx, sy, ex, ey)


if __name__ == '__main__':
    main()
