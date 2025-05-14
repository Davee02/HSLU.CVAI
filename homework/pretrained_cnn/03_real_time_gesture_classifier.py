import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import time
import os
import argparse
import ultraimport

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
            "1": "SCROLL_UP",          # Left Index Finger Up
            "2": "SCROLL_DOWN",        # Left Index Finger Down
            "3": "show_desktop",       # Left Hand Open
            "4": "play_pause",         # Left Fist
            "5": "change_volume 5",    # Left Thumb Up
            "6": "change_volume -5"    # Left Thumb Down
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

# Original gesture_names dictionary from 03_real_time_gesture_classifier.py
gesture_names = {
    "1": "Left Index Finger Up",
    "2": "Left Index Finger Down",
    "3": "Left Hand Open",
    "4": "Left Fist",
    "5": "Left Thumb Up",
    "6": "Left Thumb Down",
}

def load_model(model_path, class_info_path, device):
    """Load the trained model and class information"""
    # Load class information
    class_info = torch.load(class_info_path)
    class_names = class_info['class_names']
    
    # Determine number of classes
    num_classes = len(class_names)
    print(f"Loading model with {num_classes} classes: {class_names}")
    
    # Create model architecture (exactly matching training architecture)
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, class_names

def main(server_host="192.168.1.218", server_port=5050, consecutive_frames=10, cooldown_period=2.0):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model paths - make sure these match your saved model files
    model_path = 'model_checkpoints/best_model.pth'
    class_info_path = 'model_checkpoints/model_class_info.pth'
    
    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(class_info_path):
        print(f"Error: Model files not found at {model_path} or {class_info_path}")
        print("Make sure to train the model first!")
        return
    
    # Setup socket client
    socket_client = socket_utils.SocketClient(server_host, server_port)
    socket_client.connect()
    
    # Setup gesture controller
    controller = GestureActionController(
        socket_client=socket_client,
        consecutive_frames=consecutive_frames,  
        cooldown_period=cooldown_period
    )
    
    # Load the trained model
    model, class_names = load_model(model_path, class_info_path, device)
    
    # Define image transformations (same as validation transform used during training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    # Get initial frame to determine dimensions
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    
    # Frame dimensions
    frame_h, frame_w = frame.shape[:2]
    
    # Define the rectangle of interest (centered, 40% of frame size)
    rect_w = 200
    rect_h = 250
    rect_x = 400
    rect_y = 100
    
    # Variables for prediction timing and smoothing
    last_prediction_time = time.time() - 1.0
    prediction_cooldown = 0.2  # Time between predictions (seconds)
    
    # For prediction smoothing (optional)
    recent_predictions = []
    smoothing_window = 5

    # For FPS calculation
    prev_time = time.time()

    print("Press 'q' to quit")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time else 0
            prev_time = current_time
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Draw the rectangle of interest
            cv2.rectangle(display_frame, (rect_x, rect_y), 
                         (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)
            
            # Extract the region of interest (the hand)
            hand_roi = frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w].copy()
            
            # Make predictions at regular intervals
            current_time = time.time()
            predicted_class = None
            confidence = 0
            
            if current_time - last_prediction_time > prediction_cooldown:
                # Convert OpenCV BGR to RGB
                rgb_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and apply transformations
                pil_image = Image.fromarray(rgb_roi)
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, prediction = torch.max(probs, 1)
                    predicted_idx = prediction.item()
                    confidence = confidence.item()
                    predicted_class = class_names[predicted_idx]
                    
                    # Add to recent predictions for smoothing
                    recent_predictions.append((predicted_class, confidence))
                    if len(recent_predictions) > smoothing_window:
                        recent_predictions.pop(0)
                    
                    # Update last prediction time
                    last_prediction_time = current_time
            
            # Apply smoothing by taking the most common recent prediction
            smoothed_class = None
            avg_confidence = 0
            action_status = ""
            
            if recent_predictions:
                # Count occurrences of each class in recent predictions
                prediction_counts = {}
                for pred, conf in recent_predictions:
                    if pred not in prediction_counts:
                        prediction_counts[pred] = {"count": 0, "total_conf": 0}
                    prediction_counts[pred]["count"] += 1
                    prediction_counts[pred]["total_conf"] += conf
                
                # Find the most common prediction
                most_common = max(prediction_counts.items(), 
                                 key=lambda x: x[1]["count"])
                smoothed_class = most_common[0]
                avg_confidence = most_common[1]["total_conf"] / most_common[1]["count"]
                
                # Process the gesture for action
                action_triggered = controller.process_gesture(smoothed_class, avg_confidence)
                
                if action_triggered:
                    action_status = "Action Triggered!"
                elif controller.action_executed:
                    action_status = "Cooldown..."
                else:
                    action_status = f"Hold gesture for {controller.consecutive_frames - controller.gesture_counter} more frames"
            else:
                # No predictions yet, reset tracking
                controller.reset_tracking()
            
            # Display prediction, confidence, and gesture information
            if smoothed_class:
                # Display gesture name
                prediction_text = f"Sign: {smoothed_class} - {gesture_names[smoothed_class]} ({avg_confidence:.2f})"
                cv2.putText(display_frame, prediction_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frames held
                frames_text = f"Frames Held: {controller.gesture_counter}"
                cv2.putText(display_frame, frames_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display action status
                if action_status:
                    status_text = f"Status: {action_status}"
                    cv2.putText(display_frame, status_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display FPS
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(display_frame, fps_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the hand ROI in a separate window
            cv2.imshow('Hand ROI', hand_roi)
            
            # Show the main display
            cv2.imshow('Hand Sign Classification', display_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        # Release resources
        socket_client.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time hand gesture classification with socket integration")
    parser.add_argument("--server-host", type=str, default="192.168.1.218",
                      help="Socket server hostname")
    parser.add_argument("--server-port", type=int, default=5050,
                      help="Socket server port")
    parser.add_argument("--frames", type=int, default=10,
                      help="Number of consecutive frames before action")
    parser.add_argument("--cooldown", type=float, default=2.0,
                      help="Cooldown period between actions (seconds)")
    args = parser.parse_args()
    
    print(f"Starting real-time classifier with socket integration")
    print(f"Server: {args.server_host}:{args.server_port}")
    print(f"Consecutive frames: {args.frames}, Cooldown: {args.cooldown}s")
    
    main(
        server_host=args.server_host,
        server_port=args.server_port,
        consecutive_frames=args.frames,
        cooldown_period=args.cooldown
    )