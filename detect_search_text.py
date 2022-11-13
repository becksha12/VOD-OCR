import pytesseract

import template_matching
import video_ocr
import config
import video_utils
import cv2


def is_keyboard_present(image_gray, template, template_width, template_height):
    return template_matching.match(image_gray, template, template_width, template_height)


def extract_search_area(image):
    roi = image[config.SEARCH_BAR_MIN_Y:config.SEARCH_BAR_MAX_Y, config.SEARCH_BAR_MIN_X:config.SEARCH_BAR_MAX_X]
    return roi


def search_bar_text(image, model):
    (orig_image, o_h, o_w), (resized_image, r_h, r_w), (ratio_h, ratio_w) = video_ocr.resize_image(image, config.EAST_IMAGE_WIDTH, config.EAST_IMAGE_HEIGHT)
    boxes = video_ocr.get_bounding_boxes(resized_image, model, r_w, r_h)
    text_predictions = video_ocr.text_from_boxes(boxes, orig_image, ratio_w, ratio_h, o_w, o_h)
    return sorted(text_predictions, key=lambda x: x[0][0])


if __name__ == '__main__':
    VIDEO_PATH = './data/rec_keybrd_01.mp4'
    OUTPUT_DIR = './framewise_preds-3'
    TEMPLATE_PATH = './data/template-2-full-kbd.png'
    counter = 0
    model = video_ocr.load_east_detector(config.MODEL_PATH)
    for image in video_utils.get_frame_for_every_second(VIDEO_PATH):
        counter += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template, template_h, template_w = template_matching.read_template(TEMPLATE_PATH)
        if is_keyboard_present(gray, template, template_w, template_h):
            roi = extract_search_area(image)
            prediction = pytesseract.image_to_string(roi)
            print(counter, prediction)
