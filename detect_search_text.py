import pytesseract

import template_matching
import video_ocr
import config
import video_utils
import cv2
# import easyocr


def is_keyboard_present(image_gray, template, template_width, template_height):
    image_gray = image_gray[0:600, :]
    return template_matching.match(image_gray, template, template_width, template_height)


def extract_search_area(image):
    roi = image[config.SEARCH_BAR_MIN_Y:config.SEARCH_BAR_MAX_Y, config.SEARCH_BAR_MIN_X:config.SEARCH_BAR_MAX_X]
    return roi


def search_bar_text(image, model):
    (orig_image, o_h, o_w), (resized_image, r_h, r_w), (ratio_h, ratio_w) = video_ocr.resize_image(image, config.EAST_IMAGE_WIDTH, config.EAST_IMAGE_HEIGHT)
    boxes = video_ocr.get_bounding_boxes(resized_image, model, r_w, r_h)
    text_predictions = video_ocr.text_from_boxes(boxes, orig_image, ratio_w, ratio_h, o_w, o_h)
    return sorted(text_predictions, key=lambda x: x[0][0])


def search_bar_text_2(image):
    return pytesseract.image_to_string(image, lang='eng')


if __name__ == '__main__':
    import time
    times = {'search_bar_text': [], 'template_detection': [], 'overall': []
             }
    VIDEO_PATH = './data/rec_keybrd_01.mp4'
    OUTPUT_DIR = './framewise_preds-3'
    TEMPLATE_PATH = './data/template-2-full-kbd.png'
    counter = 0
    # reader = easyocr.Reader(['en'], gpu=False)
    # start_east = time.time()
    # model = video_ocr.load_east_detector(config.MODEL_PATH)
    # end_east = time.time()
    # print('detector load time:', end_east - start_east)
    # times['detector_load'].append(end_east - start_east)
    template, template_h, template_w = template_matching.read_template(TEMPLATE_PATH)
    for image in video_utils.get_frame_for_every_second(VIDEO_PATH):
        counter += 1
        start_pred = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        start_template = time.time()
        keyboard_found = is_keyboard_present(gray, template, template_w, template_h)
        end_template = time.time()
        if keyboard_found:
            start_search = time.time()
            roi = extract_search_area(gray)
            roi = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            prediction = pytesseract.image_to_string(roi, config='--oem 1 --psm 7')
            # prediction = reader.readtext(roi)
            end_search = time.time()
            end_pred = time.time()
            print(counter, prediction, 'template detection time:', end_template - start_template,
                  'search bar text detection time:', end_search - start_search,
                  'overall detection time:', end_pred - start_pred)
            times['template_detection'].append(end_template - start_template)
            times['search_bar_text'].append(end_search - start_search)
            times['overall'].append(end_pred - start_pred)
    print({key: sum(times[key]) / len(times[key]) for key in times})
