LAYER_NAMES = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
PADDING = 0.05
MODEL_PATH = './models/frozen_east_text_detection.pb'
IMAGE_TO_STRING_CONFIG_OPTIONS = "-l eng --oem 1 --psm 7"
EAST_IMAGE_WIDTH = 512
EAST_IMAGE_HEIGHT = 512
MIN_CONFIDENCE = 0.5

# Download the EAST model from here: https://github.com/sanifalimomin/Text-Detection-Using-OpenCV/blob/main/frozen_east_text_detection.pb