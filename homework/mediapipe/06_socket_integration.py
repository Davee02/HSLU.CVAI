import cv2
import mediapipe as mp
import torch
import time
import argparse
import ultraimport

from utils import setup_mediapipe_hands, calc_landmark_list, pre_process_landmark, setup_video_capture
from models import GestureClassifier
socket_utils = ultraimport("__dir__/../socket_utils.py")

class GestureActionController:
    def __init__(self, socket_client, consecutive_frames=10, cooldown_period=2.0):
        """
        Initialize the gesture action controller
        
        Args:
            socket_client: The socket client to send commands
            consecutive_frames: Number of consecutive frames a gesture must be detected before triggering
            cooldown_period: Time in seconds to wait between sending commands
        """
        self.socket_client = socket_client
        self.consecutive_frames = consecutive_frames
        self.cooldown_period = cooldown_period
        
        # Tracking variables
        self.current_gesture = None
        self.gesture_counter = 0
        self.last_action_time = 0
        self.action_executed = False
        
        # Define mappings for gesture labels to actions
        self.gesture_actions = {
            0: "SCROLL_UP",          # Left Index Finger Up
            1: "SCROLL_DOWN",        # Left Index Finger Down
            2: "show_desktop",       # Left Hand Open
            3: "play_pause",         # Left Fist
            4: "change_volume 5",          # Left Thumb Up
            5: "change_volume -5"         # Left Thumb Down
        }
    
    def process_gesture(self, gesture_label, confidence):
        """
        Process a detected gesture and trigger actions when appropriate
        
        Args:
            gesture_label: The recognized gesture label
            confidence: The confidence score of the detection
            
        Returns:
            action_triggered: Whether an action was triggered
        """
        current_time = time.time()
        
        # Only process high-confidence gestures
        if confidence < 0.7:
            self.reset_tracking()
            return False
        
        # Check if this is a new gesture
        if gesture_label != self.current_gesture:
            self.current_gesture = gesture_label
            self.gesture_counter = 1
            return False
        
        # Same gesture, increment counter
        self.gesture_counter += 1
        
        # Check if we've seen enough consecutive frames and cooldown period is over
        if (self.gesture_counter >= self.consecutive_frames and 
            not self.action_executed and
            current_time - self.last_action_time >= self.cooldown_period):
            
            # Get the action for this gesture
            action = self.gesture_actions.get(gesture_label)
            
            if action:
                # Send the command
                self.socket_client.send(action)
                
                # Update tracking variables
                self.last_action_time = current_time
                self.action_executed = True
                return True
        
        return False
    
    def reset_tracking(self):
        """Reset the gesture tracking"""
        self.current_gesture = None
        self.gesture_counter = 0
        self.action_executed = False


def load_model(model_path, input_size, num_classes):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def display_gesture_info(frame, gesture_label, confidence, fps, frames_held, action_status=""):
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
    info_texts = [
        f"Gesture: {gesture_name}",
        f"Confidence: {confidence:.2f}",
        f"Frames Held: {frames_held}",
        f"FPS: {fps:.1f}"
    ]
    
    if action_status:
        info_texts.append(action_status)
    
    for i, text in enumerate(info_texts):
        y_pos = 30 + (i * 30)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (8, y_pos-20), (12 + text_size[0], y_pos+5), text_bg_color, -1)
        cv2.putText(frame, text, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


def process_frame(
    frame, hands, mp_drawing, model, device, 
    controller, prev_time, last_gesture=None
):
    """Process a single frame for hand gesture recognition"""
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time
    
    # Initialize variables
    gesture_label = None
    confidence = 0.0
    action_status = ""
    
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
        
        # Process the gesture for action
        action_triggered = controller.process_gesture(gesture_label, confidence)
        
        if action_triggered:
            action_status = "Action Triggered!"
        elif controller.action_executed:
            action_status = "Cooldown..."
        else:
            action_status = f"Hold gesture for {controller.consecutive_frames - controller.gesture_counter} more frames"
    else:
        # No hand detected, reset tracking
        controller.reset_tracking()
    
    # Display gesture information
    display_gesture_info(
        frame, 
        gesture_label if gesture_label is not None else last_gesture, 
        confidence, 
        fps, 
        controller.gesture_counter if gesture_label is not None else 0,
        action_status
    )
    
    return frame, prev_time, gesture_label


def run_classifier(model_path, input_size=42, num_classes=6, server_host="DavidsPC.mshome.net", server_port=12345):
    """Run the real-time hand gesture classifier"""
    hands, mp_drawing = setup_mediapipe_hands()
    cap = setup_video_capture(camera_id=0)
    
    # Load model
    model, device = load_model(model_path, input_size, num_classes)
    print(f"Model loaded. Using device: {device}")
    
    # Setup socket client
    socket_client = socket_utils.SocketClient(server_host, server_port)
    socket_client.connect()
    
    # Setup gesture controller
    controller = GestureActionController(
        socket_client=socket_client,
        consecutive_frames=10,  # Require 10 consecutive frames of the same gesture
        cooldown_period=2.0     # 2 second cooldown between actions
    )
    
    # Initialize variables
    prev_time = 0
    last_gesture = None
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam.")
                time.sleep(0.01)
                continue
            
            frame, prev_time, current_gesture = process_frame(
                frame, hands, mp_drawing, model, device, 
                controller, prev_time, last_gesture
            )
            
            if current_gesture is not None:
                last_gesture = current_gesture
            
            cv2.imshow('Hand Gesture Classifier', frame)
            
            if cv2.waitKey(10) == 27:  # ESC to exit
                break
    
    finally:
        socket_client.close()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time hand gesture classification")
    parser.add_argument("--model", type=str, default="model_checkpoints/gesture_classifier.pth", 
                      help="Path to the trained model file")
    parser.add_argument("--input-size", type=int, default=42, 
                      help="Input size (number of features)")
    parser.add_argument("--num-classes", type=int, default=6, 
                      help="Number of classes in the model")
    parser.add_argument("--server-host", type=str, default="192.168.1.218",
                      help="Socket server hostname")
    parser.add_argument("--server-port", type=int, default=5050,
                      help="Socket server port")
    parser.add_argument("--frames", type=int, default=30,
                      help="Number of consecutive frames before action")
    parser.add_argument("--cooldown", type=float, default=2.0,
                      help="Cooldown period between actions (seconds)")
    args = parser.parse_args()
    
    print(f"Starting real-time classifier with model: {args.model}")
    print(f"Input size: {args.input_size}, Num classes: {args.num_classes}")
    print(f"Server: {args.server_host}:{args.server_port}")
    print(f"Consecutive frames: {args.frames}, Cooldown: {args.cooldown}s")
    
    run_classifier(
        model_path=args.model,
        input_size=args.input_size,
        num_classes=args.num_classes,
        server_host=args.server_host,
        server_port=args.server_port
    )