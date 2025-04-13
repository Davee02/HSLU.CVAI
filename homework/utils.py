import cv2
import mediapipe as mp
import numpy as np

def setup_video_capture(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    return cap

def setup_mediapipe_hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    mp_drawing = mp.solutions.drawing_utils
    
    return hands, mp_drawing

def calc_landmark_list(image, landmarks):
    image_height, image_width = image.shape[:2]
    
    landmark_points = []
    
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_points.append([landmark_x, landmark_y])
        
    return np.array(landmark_points)


def pre_process_landmark(landmark_list):
    base_point = landmark_list[0]
    relative_landmarks = landmark_list - base_point
    flattened = relative_landmarks.flatten()
    
    max_value = np.max(np.abs(flattened))
    if max_value > 0:
        normalized = flattened / max_value
    else:
        normalized = flattened
        
    return normalized
