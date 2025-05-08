import cv2
import numpy as np
import time
import os

def ensure_dir(directory):
    """Make sure the directory exists, create it if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    # Get initial frame to determine dimensions
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
        print("Failed to access webcam")
    
    # Frame dimensions
    frame_h, frame_w = frame.shape[:2]
    
    # Define the rectangle of interest
    rect_w = 200
    rect_h = 250
    rect_x = 400
    rect_y = 100
    
    # Create data directories
    data_dir = "data"
    ensure_dir(data_dir)
    for i in range(1, 10):
        ensure_dir(os.path.join(data_dir, str(i)))
    
    # Variables to track saved images
    save_counts = {i: 0 for i in range(1, 10)}
    last_save_time = time.time() - 1.0  # Initialize to allow immediate saving
    save_cooldown = 0.1  # Time in seconds between saves to prevent duplicates
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Draw the rectangle of interest
        cv2.rectangle(display_frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)
        
        # Extract the region of interest (the hand)
        hand_roi = frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w].copy()
        
        # Resize ROI for better visibility
        display_roi = cv2.resize(hand_roi, (frame_w, frame_h))
        
        # Create side-by-side display
        display = cv2.hconcat([display_frame, display_roi])
        
        # Add instructions text
        cv2.putText(display, "Place hand within green rectangle", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, "Press 1-9 to save hand image to corresponding folder", 
                   (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add save counts
        for i in range(1, 10):
            cv2.putText(display, f"{i}: {save_counts[i]}", 
                       (10 + (i-1)*70, display.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Resize if too large
        scale = min(1.0, 1200 / display.shape[1])
        if scale < 1.0:
            display = cv2.resize(display, (0, 0), fx=scale, fy=scale)
        
        # Show the result
        cv2.imshow('Hand Capture - Press 1-9 to save hand poses', display)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Handle number key presses (1-9)
        current_time = time.time()
        if 49 <= key <= 57:  # ASCII values for keys 1-9
            category = key - 48  # Convert ASCII to number value
            
            # Check if enough time passed since last save
            if current_time - last_save_time > save_cooldown:
                # Generate filename with timestamp and count
                save_counts[category] += 1
                filename = f"hand_pose_{time.strftime('%Y%m%d_%H%M%S')}_{save_counts[category]}.jpg"
                filepath = os.path.join(data_dir, str(category), filename)
                
                # Save the hand region
                cv2.imwrite(filepath, hand_roi)
                print(f"Saved hand pose to {filepath}")
                
                # Update last save time
                last_save_time = current_time
        
        # Exit on 'q' key press
        elif key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()