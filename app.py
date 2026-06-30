
# -*- coding: utf-8 -*-
import os
import sys
from tensorflow.keras.models import load_model
import time
# from mediapipe.python.solutions import hands as mp_hands
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'SimpleHTR', 'src')))
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from SimpleHTR.src.main import FilePaths
from SimpleHTR.src.model import Model, DecoderType
from SimpleHTR.src.dataloader_iam import Batch
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from SimpleHTR.src.preprocessor import Preprocessor
from pathlib import Path
import pyautogui as gui

from utils import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier
# from model import KeyPointClassifier
# from model import PointHistoryClassifier
import os
import tensorflow as tf
import scipy.signal
from scipy import interpolate

class AirWritingImprover:
    def __init__(self):
        self.stroke_smoothing_window = 5
        self.min_stroke_length = 8
        self.writing_guide_visible = True
        
    def smooth_stroke(self, stroke):
        """Apply smoothing to reduce jitter in air writing"""
        if len(stroke) < 3:
            return stroke
            
        # Convert to numpy arrays
        stroke_array = np.array(stroke)
        x_coords = stroke_array[:, 0]
        y_coords = stroke_array[:, 1]
        
        # Apply Savitzky-Golay filter for smoothing
        if len(stroke) >= self.stroke_smoothing_window:
            x_smooth = scipy.signal.savgol_filter(x_coords, self.stroke_smoothing_window, 2)
            y_smooth = scipy.signal.savgol_filter(y_coords, self.stroke_smoothing_window, 2)
        else:
            # Use simple moving average for short strokes
            x_smooth = np.convolve(x_coords, np.ones(3)/3, mode='same')
            y_smooth = np.convolve(y_coords, np.ones(3)/3, mode='same')
        
        return [(int(x), int(y)) for x, y in zip(x_smooth, y_smooth)]
    
    def interpolate_stroke(self, stroke, target_points=None):
        """Interpolate stroke to have consistent point density"""
        if len(stroke) < 2:
            return stroke
            
        stroke_array = np.array(stroke)
        
        # Calculate cumulative distance along stroke
        distances = np.sqrt(np.sum(np.diff(stroke_array, axis=0)**2, axis=1))
        cumulative_distance = np.concatenate(([0], np.cumsum(distances)))
        
        # Determine target number of points
        if target_points is None:
            total_length = cumulative_distance[-1]
            target_points = max(int(total_length / 5), len(stroke))  # Point every 5 pixels
        
        # Interpolate
        if len(stroke) > 2 and target_points != len(stroke):
            new_distances = np.linspace(0, cumulative_distance[-1], target_points)
            x_interp = np.interp(new_distances, cumulative_distance, stroke_array[:, 0])
            y_interp = np.interp(new_distances, cumulative_distance, stroke_array[:, 1])
            return [(int(x), int(y)) for x, y in zip(x_interp, y_interp)]
        
        return stroke
    
    def draw_writing_guide(self, image):
        """Draw guidelines to help with consistent writing"""
        height, width = image.shape[:2]
        
        # Draw baseline and guidelines
        baseline_y = int(height * 0.7)
        top_line_y = int(height * 0.4)
        
        # Main baseline (thicker)
        cv.line(image, (50, baseline_y), (width-50, baseline_y), (0, 255, 0), 2)
        
        # Top guideline
        cv.line(image, (50, top_line_y), (width-50, top_line_y), (0, 255, 0), 1)
        
        # Writing area box
        cv.rectangle(image, (50, top_line_y-20), (width-50, baseline_y+20), (0, 255, 0), 1)
        
        # Instructions
        cv.putText(image, "Write between the green lines", (60, top_line_y-30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image
air_writer = AirWritingImprover()

def distance(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
def recognize_drawing(htr_model, image_path):
    """Recognize text from the saved drawing image."""
    try:
        # Load the image
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load image: {image_path}")
            return None
            
        # Preprocess the image same way as in your infer function
        preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
        processed_img = preprocessor.process_img(img)
        
        # Create batch
        batch = Batch([processed_img], None, 1)
        
        # Run inference
        recognized, probability = htr_model.infer_batch(batch, True)
        
        return recognized[0], probability[0]
    except Exception as e:
        print(f"Error recognizing drawing: {e}")
        return None, None
# def preprocess_drawing_for_htr(image_path, target_size=(128, 32)):
#     img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#     _, thresh = cv.threshold(img, 250, 255, cv.THRESH_BINARY_INV)

#     # Find bounding box of the drawing
#     coords = cv.findNonZero(thresh)  # returns [[x, y]]
#     if coords is None:
#         return None  # nothing drawn

#     x, y, w, h = cv.boundingRect(coords)

#     # Crop to content
#     cropped = thresh[y:y+h, x:x+w]
#     pad = 10
#     cropped = cv.copyMakeBorder(cropped, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=0)

#     # Resize while keeping aspect ratio
#     aspect = w / h
#     target_w, target_h = target_size

#     if aspect > (target_w / target_h):
#         new_w = target_w
#         new_h = int(target_w / aspect)
#     else:
#         new_h = target_h
#         new_w = int(target_h * aspect)

#     resized = cv.resize(cropped, (new_w, new_h), interpolation=cv.INTER_AREA)

#     # Place resized onto white canvas
#     final = np.ones((target_h, target_w), dtype=np.uint8) * 255
#     x_offset = (target_w - new_w) // 2
#     y_offset = (target_h - new_h) // 2
#     final[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = 255 - resized  # invert back to black on white

#     return final
def preprocess_drawing_for_htr(image_path, target_size=(512, 128)):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Binarize - only if needed
    _, thresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)

    # Find content
    coords = cv.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]

    # Padding
    pad = max(w, h) // 20
    cropped = cv.copyMakeBorder(cropped, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=0)

    # Resize with better resolution (don't shrink too small)
    aspect = cropped.shape[1] / cropped.shape[0]
    target_w, target_h = target_size

    if aspect > (target_w / target_h):
        new_w = target_w
        new_h = int(target_w / aspect)
    else:
        new_h = target_h
        new_w = int(target_h * aspect)

    resized = cv.resize(cropped, (new_w, new_h), interpolation=cv.INTER_LINEAR)

    # Paste on canvas
    final = np.ones((target_h, target_w), dtype=np.uint8) * 255
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    final[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = 255 - resized  # Black on white

    return final

def get_img_height() -> int:
    """Fixed height for NN."""
    return 32
def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()

def get_img_size(line_mode: bool = False):
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()
def char_list_from_file():
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())
def infer(model: Model, fn_img):
    """Recognizes text in image provided by file path."""
    img = cv.imread(fn_img, cv.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')
args = parse_args()
decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
decoder_type = decoder_mapping[args.decoder]
HTRmodel = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
# HTRmodel=load_model('airchar_HI_model.h5')
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    n=0
    # print('we be running')
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    # print('we be capping')
    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    cv.namedWindow('Hand Gesture Recognition', cv.WINDOW_NORMAL)
    cv.setWindowProperty('Hand Gesture Recognition', cv.WND_PROP_TOPMOST, 1)   
    cv.moveWindow('Hand Gesture Recognition', 0, 0)  # Top-left corner     
    # print('we be handing')
    # Model load #############################################################
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    # mp_hands = mp.solutions.hands
    # hands = mp_hands.Hands(
    #     static_image_mode=use_static_image_mode,
    #     max_num_hands=2,
    #     min_detection_confidence=min_detection_confidence,
    #     min_tracking_confidence=min_tracking_confidence,
    # )
    # print('we be keypoint classifying')
    keypoint_classifier = KeyPointClassifier()
    # print('we be history')
    point_history_classifier = PointHistoryClassifier()
    # print('we be opening')
    # Read labels ###########################################################
    with open('D:/AIML/tonystark/hand-gesture-recognition-mediapipe/model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'D:/AIML/tonystark/hand-gesture-recognition-mediapipe/model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    drawrect = []          # current stroke
    all_strokes = []       # completed strokes
    drawing_mode = False 
    while True:
        # print('we be looping')
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)

        # OpenCV -> RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Create MediaPipe Image (NO deepcopy)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Detect (for webcam/video)
        timestamp_ms = int(time.time() * 1000)
        results = detector.detect(mp_image)
        debug_image=copy.deepcopy(frame)

        # Use landmarks
        if results.hand_landmarks:
            for hand_landmarks, handedness in zip(results.hand_landmarks,
                                                  results.handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_label = handedness[0].category_name

                # Because we're flipping the webcam
                if hand_label == "Left":
                    hand_label = "Right"
                else:
                    hand_label = "Left"
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_label == "Right" : 
                    if hand_sign_id == 2: 
                        curr_point = landmark_list[8]
                        if len(drawrect) == 0 or distance(curr_point, drawrect[-1]) > 21:
                            drawrect.append(curr_point)
                        # drawrect.append(landmark_list[8])
                    if hand_sign_id==0:
                        if len(drawrect) > 3:  # Only add if there's something to add
                            all_strokes.append(drawrect)
                            drawrect = []
                if air_writer.writing_guide_visible:
                     debug_image = air_writer.draw_writing_guide(debug_image)
                
                if hand_sign_id==1 and hand_label=="Left":
                    if len(drawrect)>4:
                        all_strokes.append(drawrect)
                        drawrect = []
                    if len(all_strokes)>0:
                        canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255
                        # for i in range(1, len(drawrect)):
                        #     cv.line(canvas, drawrect[i - 1], drawrect[i], (0, 0, 0), thickness=2)
                        for stroke in all_strokes:
                            for i in range(1,len(stroke)):
                                pt1 = tuple(stroke[i - 1])
                                pt2 = tuple(stroke[i])
                                cv.line(canvas, pt1, pt2, (0, 0, 0), thickness=11)
                        cv.imwrite('drawing.jpg', canvas)
                        preprocessed_img = preprocess_drawing_for_htr("drawing.jpg")
                        if preprocessed_img is not None:
                                cv.imwrite(f"./data/I/I_{n:03}.jpg", preprocessed_img)
                                n+=1
                        if n==30:
                            print("meow meow hogya oyeeeee")
                        if preprocessed_img is not None:
                                recognized, probability = recognize_drawing(
    HTRmodel,
    "drawing_preprocessed.jpg"
)
                                label = 'H' if prediction[0] > prediction[1] else 'I'
                                print(f"Prediction: {label} ({prediction})")
                        recognized_text, probability = recognize_drawing(HTRmodel, "drawing_preprocessed.jpg")
                        if recognized_text is not None:
                            print(f"Recognized text: '{recognized_text}'")
                            print(f"Confidence: {probability:.4f}")
                            gui.write(recognized_text + ' ', interval=0.05)

                        else:
                            print("Failed to recognize text")
                        all_strokes=[]
                if hand_sign_id==3 and hand_label=="Left":
                    # if len(drawrect)>4:
                    #     all_strokes.append(drawrect)
                    #     drawrect = []
                    # if len(all_strokes)>0:
                    #     canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255
                    #     # for i in range(1, len(drawrect)):
                    #     #     cv.line(canvas, drawrect[i - 1], drawrect[i], (0, 0, 0), thickness=2)
                    #     for stroke in all_strokes:
                    #         for i in range(1,len(stroke)):
                    #             pt1 = tuple(stroke[i - 1])
                    #             pt2 = tuple(stroke[i])
                    #             cv.line(canvas, pt1, pt2, (0, 0, 0), thickness=11)
                    #     cv.imwrite('drawing.jpg', canvas)
                    #     preprocessed_img = preprocess_drawing_for_htr("drawing.jpg")
                    #     if preprocessed_img is not None:
                    #             cv.imwrite(f"H_{n:03}.jpg", preprocessed_img)
                    #             n+=1
                    #     if n==30:
                    #         print("meow meow hogya oyeeeee")
                    #     # recognized_text, probability = recognize_drawing(HTRmodel, "drawing_preprocessed.jpg")
                    #     # if recognized_text is not None:
                    #     #     print(f"Recognized text: '{recognized_text}'")
                    #     #     print(f"Confidence: {probability:.4f}")
                    #     #     gui.write(recognized_text + ' ', interval=0.05)

                    #     else:
                    #         print("Failed to recognize text")
                        all_strokes=[]
                        drawrect=[]
                # if hand_sign_id==0 and handedness.classification[0].label=="Left":
                #     gui.press('enter')
                

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        for stroke in all_strokes:
            if len(stroke) > 4:
                for i in range(1, len(stroke)):
                    pt1 = tuple(stroke[i - 1])
                    pt2 = tuple(stroke[i])
                    cv.line(debug_image, pt1, pt2, (0, 0, 0), thickness=11)
        if len(drawrect) > 4:
                for i in range(1, len(drawrect)):
                    pt1 = tuple(drawrect[i - 1])
                    pt2 = tuple(drawrect[i])
                    cv.line(debug_image, pt1, pt2, (0, 150, 0), thickness=11) 
            # if len(drawrect)>4:
            #             # for i in range(1, len(drawrect)):
            #             #     cv.line(canvas, drawrect[i - 1], drawrect[i], (0, 0, 0), thickness=2)
            #             for i in range(1,len(drawrect)):
            #                     pt1 = tuple(drawrect[i - 1])
            #                     pt2 = tuple(drawrect[i])
            #                     cv.line(debug_image, pt1, pt2, (0, 0, 0), thickness=2)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    hand_label = handedness[0].category_name

# Because we're flipping the webcam
    if hand_label == "Left":
        hand_label = "Right"
    else:
        hand_label = "Left"
    info_text = hand_label
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()
