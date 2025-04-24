import cv2
import numpy as np
import time
import os

def extract_hand_by_skin_color(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color in HSV (adjusted for better detection)
    lower_skin = np.array([0, 50, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a binary mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Noise removal with morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)
    
    # Extract hand region
    hand_region = cv2.bitwise_and(image, image, mask=mask)
    return hand_region, mask

class HandTracker:
    def __init__(self):
        self.prev_contour = None
        self.prev_box = None
        self.box_history = []
        self.last_valid_time = 0
        self.min_contour_area = 3000  # Minimum contour area to consider
        self.stability_frames = 10     # Number of frames for averaging
        
    def track_hand(self, image, mask):
        result = image.copy()
        cropped_hand = None
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_contour = None
        largest_area = 0
        
        # Find the largest contour that meets the minimum area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area and area > largest_area:
                largest_area = area
                largest_contour = contour
        
        current_time = time.time()
        box = None
        
        if largest_contour is not None:
            # Update the last valid detection time
            self.last_valid_time = current_time
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            box = (x, y, w, h)
            
            # Add to history
            self.box_history.append(box)
            if len(self.box_history) > self.stability_frames:
                self.box_history.pop(0)
            
            # Draw the contour
            cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)
            
            # Update the previous contour
            self.prev_contour = largest_contour
        
        # If no valid contour is found but we had one recently (within 0.5 seconds)
        elif self.prev_contour is not None and current_time - self.last_valid_time < 0.5:
            # Draw the previous contour in yellow to indicate it's being held
            cv2.drawContours(result, [self.prev_contour], -1, (0, 255, 255), 2)
            
            # Use the previous box
            if self.prev_box is not None:
                box = self.prev_box
            
        # Calculate smoothed bounding box if we have history
        if len(self.box_history) > 0:
            # Calculate average box coordinates from history
            avg_x = sum(b[0] for b in self.box_history) / len(self.box_history)
            avg_y = sum(b[1] for b in self.box_history) / len(self.box_history)
            avg_w = sum(b[2] for b in self.box_history) / len(self.box_history)
            avg_h = sum(b[3] for b in self.box_history) / len(self.box_history)
            
            # Use the smoothed box
            smooth_box = (int(avg_x), int(avg_y), int(avg_w), int(avg_h))
            
            # Draw the smoothed bounding rectangle
            x, y, w, h = smooth_box
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract the hand region
            if x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= image.shape[1] and y+h <= image.shape[0]:
                cropped_hand = image[y:y+h, x:x+w].copy()
            
            # Update the previous box
            self.prev_box = smooth_box
            
        return result, cropped_hand

def ensure_dir(directory):
    """Make sure the directory exists, create it if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    # Create hand tracker
    tracker = HandTracker()
    
    # Create data directories
    data_dir = "data"
    ensure_dir(data_dir)
    for i in range(1, 10):
        ensure_dir(os.path.join(data_dir, str(i)))
    
    # Variables to track saved images
    save_counts = {i: 0 for i in range(1, 10)}
    last_save_time = time.time() - 1.0  # Initialize to allow immediate saving
    save_cooldown = 0.5  # Time in seconds between saves to prevent duplicates
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Extract hand region
        hand_region, mask = extract_hand_by_skin_color(frame)
        
        # Track hand and get cropped hand
        result, cropped_hand = tracker.track_hand(frame, mask)
        
        # Create a composite display
        frame_h, frame_w = frame.shape[:2]
        
        # Top row: original and mask
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        top_row = cv2.hconcat([frame, mask_bgr])
        
        # Middle row: hand region and result
        middle_row = cv2.hconcat([hand_region, result])
        
        # Make sure top and middle rows have same width
        if top_row.shape[1] != middle_row.shape[1]:
            middle_row = cv2.resize(middle_row, (top_row.shape[1], middle_row.shape[0]))
        
        # Bottom row: cropped hand (centered)
        if cropped_hand is None or cropped_hand.size == 0:
            bottom_row = np.zeros((frame_h, top_row.shape[1], 3), dtype=np.uint8)
            # Add text indicating no hand detected
            cv2.putText(bottom_row, "No hand detected", (bottom_row.shape[1]//2 - 100, bottom_row.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            has_valid_hand = False
        else:
            has_valid_hand = True
            # Create a blank canvas the same width as top_row
            bottom_row = np.zeros((frame_h, top_row.shape[1], 3), dtype=np.uint8)
            
            # Resize cropped hand to fit while maintaining aspect ratio
            h, w = cropped_hand.shape[:2]
            scale = min(frame_w / w, frame_h / h) * 0.8  # 80% of max possible size
            new_w, new_h = int(w * scale), int(h * scale)
            
            if new_w > 0 and new_h > 0:  # Make sure we have valid dimensions
                resized_hand = cv2.resize(cropped_hand, (new_w, new_h))
                
                # Calculate position to center the cropped hand
                x_offset = (bottom_row.shape[1] - new_w) // 2
                y_offset = (bottom_row.shape[0] - new_h) // 2
                
                # Place the cropped hand on the background
                bottom_row[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_hand
        
        # Make sure all rows have the same width
        if bottom_row.shape[1] != top_row.shape[1]:
            bottom_row = cv2.resize(bottom_row, (top_row.shape[1], bottom_row.shape[0]))
        
        # Combine all rows vertically
        display = cv2.vconcat([top_row, middle_row, bottom_row])
        
        # Add save instructions text
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
        cv2.imshow('Hand Detection - Press 1-9 to save hand poses', display)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Handle number key presses (1-9)
        current_time = time.time()
        if 49 <= key <= 57:  # ASCII values for keys 1-9
            category = key - 48  # Convert ASCII to number value
            
            # Only save if we have a valid hand and enough time passed since last save
            if has_valid_hand and cropped_hand is not None and current_time - last_save_time > save_cooldown:
                # Generate filename with timestamp and count
                save_counts[category] += 1
                filename = f"hand_pose_{time.strftime('%Y%m%d_%H%M%S')}_{save_counts[category]}.jpg"
                filepath = os.path.join(data_dir, str(category), filename)
                
                # Save the cropped hand image
                cv2.imwrite(filepath, cropped_hand)
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