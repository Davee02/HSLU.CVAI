import cv2
import mediapipe as mp
import torch
import time

from utils import setup_mediapipe_hands, calc_landmark_list, pre_process_landmark, setup_video_capture
from models import GestureClassifier


def load_model(model_path, input_size, num_classes):
    """Load the trained model"""
    device = torch.device('cuda')
    model = GestureClassifier(input_size=input_size, num_classes=num_classes, hidden_size=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def classify_landmarks(landmarks, model, device):
    """Classify the hand landmarks using the trained model"""
    with torch.no_grad():
        tensor = torch.tensor(landmarks, dtype=torch.float).to(device)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    return predicted_class, confidence


def display_gesture_info(frame, gesture_label, confidence, fps):
    """Display gesture information on the frame"""
    # Map the numerical label to gesture name (for better user understanding)
    gesture_names = {
        0: "Left Index Finger Up",
        1: "Left Index Finger Down",
        2: "Left Hand Open",
        3: "Left Fist",
        4: "Left Thumb Up",
        5: "Left Thumb Down",
    }
    
    # Background for text
    text_bg_color = (0, 0, 0)
    text_color = (0, 255, 0)
    
    # Get gesture name
    gesture_name = gesture_names.get(gesture_label, f"Gesture {gesture_label}")
    
    # Display gesture and confidence with background
    for i, text in enumerate([
        f"Gesture: {gesture_name}",
        f"Confidence: {confidence:.2f}",
        f"FPS: {fps:.1f}"
    ]):
        y_pos = 30 + (i * 30)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (8, y_pos-20), (12 + text_size[0], y_pos+5), text_bg_color, -1)
        cv2.putText(frame, text, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


def process_frame(
    frame, hands, mp_drawing, model, device, 
    prev_time, gesture_label=None, confidence=0.0, fps=0.0
):
    """Process a single frame for hand gesture recognition"""
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time
    
    # Flip and convert frame for MediaPipe
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Draw hand landmarks and classify gesture
    if results.multi_hand_landmarks:
        hand_landmark = results.multi_hand_landmarks[0]
        
        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame, 
            hand_landmark, 
            mp.solutions.hands.HAND_CONNECTIONS
        )
        
        # Extract and preprocess landmarks
        landmarks = calc_landmark_list(frame, hand_landmark)
        normalized_landmarks = pre_process_landmark(landmarks)
        
        # Classify the gesture
        gesture_label, confidence = classify_landmarks(normalized_landmarks, model, device)
    
        # Display gesture information
        display_gesture_info(frame, gesture_label, confidence, fps)
    
    return frame, prev_time, gesture_label, confidence, fps


def run_classifier(model_path, input_size=42, num_classes=2):
    """Run the real-time hand gesture classifier"""
    hands, mp_drawing = setup_mediapipe_hands()
    cap = setup_video_capture(camera_id=0)
    
    # Load model
    model, device = load_model(model_path, input_size, num_classes)
    print(f"Model loaded. Using device: {device}")
    
    # Initialize variables
    prev_time = 0
    gesture_label = None
    confidence = 0.0
    fps = 0.0
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam.")
                time.sleep(0.01)
                continue
            
            frame, prev_time, gesture_label, confidence, fps = process_frame(
                frame, hands, mp_drawing, model, device, 
                prev_time, gesture_label, confidence, fps
            )
            
            cv2.imshow('Hand Gesture Classifier', frame)
            
            if cv2.waitKey(10) == 27:  # ESC to exit
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time hand gesture classification")
    parser.add_argument("--model", type=str, default="model_checkpoints/gesture_classifier.pth", 
                        help="Path to the trained model file")
    parser.add_argument("--input-size", type=int, default=42, 
                        help="Input size (number of features)")
    parser.add_argument("--num-classes", type=int, default=4, 
                        help="Number of classes in the model")
    args = parser.parse_args()
    
    print(f"Starting real-time classifier with model: {args.model}")
    print(f"Input size: {args.input_size}, Num classes: {args.num_classes}")
    
    run_classifier(
        model_path=args.model,
        input_size=args.input_size,  # 21 landmarks with x,y coordinates
        num_classes=args.num_classes,  # 0: Finger Down, 1: Finger Up
    )