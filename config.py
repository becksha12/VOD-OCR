LAYER_NAMES = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
PADDING = 0.05
MODEL_PATH = './models/frozen_east_text_detection_latest.pb'
IMAGE_TO_STRING_CONFIG_OPTIONS = "-l eng --oem 1 --psm 7"
EAST_IMAGE_WIDTH = 512
EAST_IMAGE_HEIGHT = 512
MIN_CONFIDENCE = 0.2
KEYBOARD_TEMPLATE_CORR_THRESHOLD = 10.5
SEARCH_BAR_MIN_X = 146
SEARCH_BAR_MIN_Y = 48
SEARCH_BAR_MAX_X = 1834
SEARCH_BAR_MAX_Y = 98


KEYBOARD_MIN_X = 120
KEYBOARD_MIN_Y = 135
KEYBOARD_MAX_X = 1900
KEYBOARD_MAX_Y = 400

# Download the EAST model from hetre: https://github.com/sanifalimomin/Text-Detection-Using-OpenCV/blob/main/frozen_east_text_detection.pb
