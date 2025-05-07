import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import time
import os

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

def main():
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
    if not ret:
        print("Failed to access webcam")
        return
    
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

    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            continue
        
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
            
            # Display prediction and confidence
            prediction_text = f"Sign: {smoothed_class} ({avg_confidence:.2f})"
            cv2.putText(display_frame, prediction_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show the hand ROI in a separate window
        cv2.imshow('Hand ROI', hand_roi)
        
        # Show the main display
        cv2.imshow('Hand Sign Classification', display_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()