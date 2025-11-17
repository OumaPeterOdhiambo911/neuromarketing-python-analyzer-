# Python Facial Emotion Analyzer

This tool analyzes a video file to detect facial emotions and generates:
1.  An analyzed video with emotion overlays.
2.  A JSON file with time-stamped emotional data.
3.  An interactive HTML graph of the "Emotional Arc".

## Setup

1.  Clone the repository.
2.  Install dependencies:
    `pip install -r requirements.txt`

## Usage

Place your video (e.g., `input_ad.mp4`) in the root folder.

`python analyze_ad.py`

Or, specify a path:

`python analyze_ad.py "path/to/your/video.mp4"`

import cv2
import json
import pandas as pd
import plotly.graph_objects as go
from fer import FER
import os
import sys
import time

def analyze_video(input_video_path, process_every_n_frames=5):
    """
    Analyzes a video file for facial emotions and generates three deliverables:
    1. An output video with emotions overlaid.
    2. A JSON file with time-stamped emotion data.
    3. An interactive HTML graph of the "Emotional Arc".
    """
    
    # --- 1. INITIALIZATION ---
    
    print(f"Starting analysis for: {input_video_path}")
    
    # Define output file paths
    base_name = os.path.splitext(input_video_path)[0]
    output_video_path = f"{base_name}_analyzed.mp4"
    json_output_path = f"{base_name}_emotional_arc.json"
    graph_output_path = f"{base_name}_emotional_arc.html"
    
    # Initialize the FER detector
    # Using mtcnn=True is more accurate but slower.
    # Set to False to use OpenCV's faster Haar cascades.
    try:
        detector = FER(mtcnn=True)
        print("FER (MTCNN) detector initialized.")
    except Exception as e:
        print(f"Error initializing FER detector: {e}")
        print("Please ensure TensorFlow is correctly installed.")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    # Get video properties for the output writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0:
        print("Warning: Could not read FPS. Defaulting to 25.")
        fps = 25 # Set a default if FPS is not available

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    print(f"Output video will be saved to: {output_video_path}")

    # Data storage
    all_emotions_data = [] # For the JSON file
    last_known_faces = []  # To persist drawings on skipped frames
    frame_number = 0
    start_time = time.time()

    # --- 2. VIDEO PROCESSING LOOP ---

    print("Processing video frame by frame...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Calculate timestamp
        timestamp_sec = frame_number / fps
        
        # --- Core Task: Detect & Analyze Emotions ---
        # We process every Nth frame to speed things up.
        # The drawings from `last_known_faces` will be applied to every frame.
        if frame_number % process_every_n_frames == 0:
            try:
                # `detect_emotions` returns a list of face dictionaries
                # Each dict: {'box': [x, y, w, h], 'emotions': {'happy': 0.9, ...}}
                last_known_faces = detector.detect_emotions(frame)
                
                # --- Store data for JSON (Deliverable 2) ---
                if last_known_faces:
                    frame_data_for_json = {
                        'timestamp_sec': timestamp_sec,
                        'timestamp_str': f"{int(timestamp_sec // 60):02}:{int(timestamp_sec % 60):02}.{int((timestamp_sec % 1) * 100)}",
                        'faces': []
                    }
                    
                    for face in last_known_faces:
                        # Convert numpy types to standard int/float for JSON
                        face_data = {
                            'box': [int(b) for b in face['box']],
                            'emotions': {e: float(p) for e, p in face['emotions'].items()},
                            'dominant_emotion': max(face['emotions'], key=face['emotions'].get)
                        }
                        frame_data_for_json['faces'].append(face_data)
                    
                    all_emotions_data.append(frame_data_for_json)
                    
            except Exception as e:
                # Catch errors (e.g., from MTCNN) and continue
                print(f"Error during emotion detection on frame {frame_number}: {e}")
                last_known_faces = [] # Clear faces if detection fails

        # --- Draw on Output Frame (Deliverable 1) ---
        # We draw on *every* frame, using the last known face data
        # This makes the output video look smooth, not "flickery"
        if last_known_faces:
            for face in last_known_faces:
                (x, y, w, h) = face['box']
                emotions = face['emotions']
                dominant_emotion = max(emotions, key=emotions.get)
                dominant_prob = emotions[dominant_emotion]

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Create text for dominant emotion
                text = f"{dominant_emotion}: {dominant_prob:.2f}"
                
                # Draw a semi-transparent background for the text
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y - 5), (0, 100, 0), -1)
                
                # Draw the text
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write the frame to the output video
        out.write(frame)

        # Optional: Show a preview window
        # cv2.imshow('Neuromarketing Analysis', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        frame_number += 1
        
        # Print progress
        if frame_number % (int(fps) * 5) == 0: # Every 5 seconds of video
             elapsed = time.time() - start_time
             print(f"  Processed {timestamp_sec:.2f}s of video... (Real time elapsed: {elapsed:.2f}s)")

    # --- 3. CLEANUP & FILE GENERATION ---
    
    print("\nVideo processing complete.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")
    print(f"Analyzed video saved to: {output_video_path}")

    # --- Deliverable 2: Save JSON File ---
    if not all_emotions_data:
        print("Warning: No faces were detected in the video. Skipping JSON and graph output.")
        return

    with open(json_output_path, 'w') as f:
        json.dump(all_emotions_data, f, indent=4)
    print(f"Emotional data saved to: {json_output_path}")

    # --- Deliverable 3: Create "Emotional Arc" Graph ---
    try:
        print("Generating emotional arc graph...")
        # Flatten the data for Plotly, assuming one primary face (the first detected)
        # For a more complex analysis, you could average all faces or track specific faces.
        plot_data = []
        for frame_data in all_emotions_data:
            if frame_data['faces']:
                primary_face_emotions = frame_data['faces'][0]['emotions']
                row = {
                    'timestamp': frame_data['timestamp_sec'],
                    **primary_face_emotions
                }
                plot_data.append(row)

        if not plot_data:
             print("No face data was extracted for plotting.")
             return

        df = pd.DataFrame(plot_data)
        df = df.set_index('timestamp')

        # Smooth the data with a rolling average to reduce noise
        # We use a 1-second rolling window
        rolling_window_size = int(fps / process_every_n_frames)
        if len(df) > rolling_window_size:
            df_smoothed = df.rolling(window=rolling_window_size, min_periods=1).mean()
        else:
            df_smoothed = df # Not enough data to smooth
            print("Warning: Not enough data for full 1-second smoothing.")

        # Create the Plotly figure
        fig = go.Figure()
        emotions_to_plot = ['happy', 'sad', 'angry', 'surprise', 'neutral', 'fear', 'disgust']
        
        for emotion in emotions_to_plot:
            if emotion in df_smoothed.columns:
                fig.add_trace(go.Scatter(
                    x=df_smoothed.index,
                    y=df_smoothed[emotion],
                    mode='lines',
                    name=emotion.capitalize(),
                    hovertemplate=f"<b>{emotion.capitalize()}</b>: %{{y:.2f}}<br><b>Time</b>: %{{x:.2f}}s"
                ))
        
        fig.update_layout(
            title='Emotional Arc of the Advertisement (Smoothed)',
            xaxis_title='Time (seconds)',
            yaxis_title='Emotion Probability (0.0 to 1.0)',
            legend_title='Emotions',
            hovermode="x unified",
            template="plotly_dark"
        )
        
        # Save as an interactive HTML file
        fig.write_html(graph_output_path)
        print(f"Interactive graph saved to: {graph_output_path}")
        print("\n--- Analysis Complete ---")

    except Exception as e:
        print(f"Error during graph generation: {e}")

# --- 4. SCRIPT EXECUTION ---

if __name__ == "__main__":
    # Get video path from command line argument or use default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'input_ad.mp4' # Default video name

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        print("Usage: python analyze_ad.py [path_to_your_video.mp4]")
    else:
        # Adjust 'process_every_n_frames' based on your machine's power
        # Lower number = More accurate but MUCH slower
        # Higher number = Faster processing but less granular data
        analyze_video(video_path, process_every_n_frames=5)
