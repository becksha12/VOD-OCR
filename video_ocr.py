import numpy as np
import cv2
import pytesseract
import math
import config
from imutils.object_detection import non_max_suppression


def decode_predictions(scores, geometry, min_confidence):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scores_data[x] < min_confidence:
                continue
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            end_x = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])
    # return a tuple of the bounding boxes and associated confidences
    return rects, confidences

def get_text(image):
    return pytesseract.image_to_string(image)


def load_east_detector(model_path):
    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(model_path)
    return net


def resize_image(image, width, height):
    # load the input image and grab the image dimensions
    orig = image.copy()
    (orig_h, orig_w) = image.shape[:2]
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (new_w, new_h) = (width, height)
    r_w = orig_w / float(new_w)
    r_h = orig_h / float(new_h)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (new_w, new_h))
    (H, W) = image.shape[:2]
    return (orig, orig_h, orig_w), (image, H, W), (r_h, r_w)


def get_bounding_boxes(image, net, width, height):
    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(config.LAYER_NAMES)
    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry, config.MIN_CONFIDENCE)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    return boxes


def text_from_boxes(boxes, original_image, ratio_w, ratio_h, orig_w, orig_h):
    results = []
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * ratio_w)
        startY = int(startY * ratio_h)
        endX = int(endX * ratio_w)
        endY = int(endY * ratio_h)
        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * config.PADDING)
        dY = int((endY - startY) * config.PADDING)
        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(orig_w, endX + (dX * 2))
        endY = min(orig_h, endY + (dY * 2))
        # extract the actual padded ROI
        roi = original_image[startY:endY, startX:endX]
        text = pytesseract.image_to_string(roi, config=config.IMAGE_TO_STRING_CONFIG_OPTIONS)
        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))
    return results


def display_bounding_boxes(predictions, image, to_display_text=True, save_path='output.png'):
    output = image.copy()
    for ((startX, startY, endX, endY), text) in predictions:
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(output, (startX, startY), (endX, endY),
                      (0, 0, 255), 4)
        if to_display_text:
            cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    # show the output image
    cv2.imwrite(save_path, output)


def main():
    FILE_PATH = './data/beast_peacock-ss_1.png'
    frame_counter = 0

    # for image in get_frame_for_every_second(FILE_PATH):
    #     text = get_text(image)
    #     print('Frame #', frame_counter)
    #     print(text)
    #     frame_counter += 1

    model = load_east_detector(config.MODEL_PATH)
    image = cv2.imread(FILE_PATH)
    (orig_image, o_h, o_w), (resized_image, r_h, r_w), (ratio_h, ratio_w) = resize_image(image, config.EAST_IMAGE_WIDTH, config.EAST_IMAGE_HEIGHT)
    boxes = get_bounding_boxes(resized_image, model, r_w, r_h)
    text_predictions = text_from_boxes(boxes, orig_image, ratio_w, ratio_h, o_w, o_h)
    print(len(text_predictions))
    print(text_predictions)
    display_bounding_boxes(text_predictions, orig_image, False)


if __name__ == '__main__':
    main()
