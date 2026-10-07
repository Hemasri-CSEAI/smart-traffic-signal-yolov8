import cv2
import time
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# --- STEP 1: INITIALIZE AI MODEL ---
print("Initializing AI components...")
model = YOLO('yolov8n.pt')

# --- STEP 2: OPEN A FILE WINDOW TO CHOOSE YOUR VIDEO ---
root = tk.Tk()
root.withdraw()  
root.attributes('-topmost', True)  

print("Opening File Selector Window...")
selected_file = filedialog.askopenfilename(
    title="Select a Traffic Video Feed for AI Analysis",
    filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
)

if not selected_file:
    print("No file selected. Exiting system.")
    exit()

video_name = os.path.basename(selected_file)

# --- STEP 3: SETUP SIGNAL VARIABLES ---
cap = cv2.VideoCapture(selected_file)

BASE_TIME = 10        
TIME_PER_UNIT = 2     
MAX_LIMIT = 60        

is_lane_a_green = True
start_time = time.time()
allocated_time = 15   

# Unlock window borders so it can scale without cropping
cv2.namedWindow("Multi-Feed Smart Traffic System Prototype", cv2.WINDOW_NORMAL)

print(f"\nSuccessfully loaded: {video_name}")
print("Processing started. Press 'q' on the video screen to close.")

# --- STEP 4: MAIN PROCESSING LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # --- 4A. RESIZE THE FRAME FIRST ---
    # Resizing the frame BEFORE adding text fixes the tiny font problem!
    resized_frame = cv2.resize(frame, (850, 480))

    # --- 4B. RUN AI ON RESIZED FRAME ---
    results = model(resized_frame, classes=[2, 3, 5, 7], verbose=False, conf=0.4)
    vehicle_count = len(results[0].boxes)

    # Dynamic Allocation Logic
    elapsed = time.time() - start_time
    remaining = max(0, int(allocated_time - elapsed))

    if remaining <= 0:
        is_lane_a_green = not is_lane_a_green
        start_time = time.time()
        allocated_time = min(MAX_LIMIT, BASE_TIME + (vehicle_count * TIME_PER_UNIT))

    # --- 4C. CLEAN INTERACTIVE DASHBOARD ---
    # Plot thin bounding boxes on our resized frame
    annotated_frame = results[0].plot(labels=False, conf=False, line_width=1)
    
    # Create a perfectly proportioned black banner background at the top (Height: 110px)
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, 0), (850, 110), (15, 15, 15), -1)
    annotated_frame = cv2.addWeighted(overlay, 0.85, annotated_frame, 0.15, 0)

    # --- 4D. ADD LARGE, PRESENTATION-READY TEXT ---
    # Source Info (Small text at the very top edge)
    cv2.putText(annotated_frame, f"FEED: {video_name.upper()}", (15, 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Lane A Status (Left side)
    a_color = (0, 255, 0) if is_lane_a_green else (0, 0, 255)
    a_text = f"LANE A: GREEN ({remaining}s)" if is_lane_a_green else "LANE A: RED (WAITING)"
    cv2.putText(annotated_frame, a_text, (15, 55), 
                cv2.FONT_HERSHEY_DUPLEX, 0.65, a_color, 2)

    # Lane B Status (Left side, slightly lower)
    b_color = (0, 255, 0) if not is_lane_a_green else (0, 0, 255)
    b_text = f"LANE B: GREEN ({remaining}s)" if not is_lane_a_green else "LANE B: RED (WAITING)"
    cv2.putText(annotated_frame, b_text, (15, 90), 
                cv2.FONT_HERSHEY_DUPLEX, 0.65, b_color, 2)

    # Live Density Data (Right side)
    cv2.putText(annotated_frame, f"LIVE DENSITY: {vehicle_count} UNITS", (480, 55), 
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
    
    # Next Allocation Calculation (Right side, slightly lower)
    cv2.putText(annotated_frame, f"NEXT ALLOC: {int(allocated_time)}s", (480, 90), 
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 215, 255), 2)

    # --- 4E. DISPLAY THE FINAL RENDER ---
    cv2.imshow("Multi-Feed Smart Traffic System Prototype", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Execution ended cleanly.")