import cv2
import mediapipe as mp
import os
import pandas as pd
import time
from utils import calc_landmark_list, pre_process_landmark, setup_mediapipe_hands, setup_video_capture

def load_existing_data(file_path):
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_parquet(file_path)
            records = existing_df.to_dict('records')
            print(f"Loaded {len(records)} existing landmark records")
            return records
        except Exception as e:
            print(f"Error loading existing file: {e}")
    
    return []

def save_landmarks(landmarks, label, all_data, file_path):
    record = {"label": label, "timestamp": time.time()}
    
    for i, value in enumerate(landmarks):
        record[f"landmark_{i}"] = float(value)
    
    all_data.append(record)
    
    df = pd.DataFrame(all_data)
    df.to_parquet(file_path, index=False)
    
    print(f"Saved landmarks with label {label}. Total records: {len(all_data)}")
    
    return all_data


def process_frame(frame, hands, mp_drawing, clicked_number, all_data, file_path):
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    status_text = f"Records: {len(all_data)} | Press 0-9 to save hand pose"
    cv2.putText(frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if results.multi_hand_landmarks:
        hand_landmark = results.multi_hand_landmarks[0]
        
        mp_drawing.draw_landmarks(
            frame, 
            hand_landmark, 
            mp.solutions.hands.HAND_CONNECTIONS
        )
        
        if clicked_number != -1:
            landmarks = calc_landmark_list(frame, hand_landmark)
            normalized_landmarks = pre_process_landmark(landmarks)
            
            all_data = save_landmarks(normalized_landmarks, clicked_number, all_data, file_path)
            
            cv2.putText(frame, f"Saved label {clicked_number}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            clicked_number = -1
    
    return frame, clicked_number, all_data


def run_capture_loop(cap, hands, mp_drawing, all_data, file_path):
    clicked_number = -1
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam.")
            time.sleep(0.01)
            continue
        
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        elif 48 <= key <= 57:  # 0-9
            clicked_number = key - 48
            print(f"Clicked number: {clicked_number}")
        
        frame, clicked_number, all_data = process_frame(
            frame, hands, mp_drawing, clicked_number, all_data, file_path
        )
        
        cv2.imshow('Hand Gesture Recognition', frame)
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_parquet(file_path, index=False)
        print(f"Final save complete. Total records: {len(all_data)}")


def cleanup_resources(cap, hands):
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


def main():
    output_file = 'hand_landmarks.parquet'
    camera_id = 0
    
    hands, mp_drawing = setup_mediapipe_hands()
    cap = setup_video_capture(camera_id)
    all_data = load_existing_data(output_file)
    
    try:
        run_capture_loop(cap, hands, mp_drawing, all_data, output_file)
    finally:
        cleanup_resources(cap, hands)


if __name__ == "__main__":
    main()