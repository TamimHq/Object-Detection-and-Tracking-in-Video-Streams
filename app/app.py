import streamlit as st
import cv2
import csv
import os
import tempfile
import subprocess
from ultralytics import YOLO
import imageio_ffmpeg as imageio_ff

# --- APP CONFIGURATION ---
st.set_page_config(page_title="YOLO Object Counter", layout="wide")
st.title("🚗 YOLO Video Object Counter")
st.write("Upload a video to track and count objects passing through the center zone.")
st.info("💡 **Note:** This app runs on a standard cloud CPU. Processing may take a few minutes. For testing, please upload short clips (5-10 seconds)!")

# --- LOAD MODEL ---
# Using st.cache_resource ensures the model only loads once per session
@st.cache_resource
def load_model():
    return YOLO("best.pt") 

model = load_model()

# --- FILE UPLOAD ---
uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'mov', 'avi'])

if uploaded_video is not None:
    st.video(uploaded_video) # Show original video
    
    if st.button("Start Processing Video"):
        with st.spinner("Processing video... This may take a few minutes."):
            
            # 1. Save uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                video_path = tfile.name
            
            # 2. Setup output paths
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t1:
                raw_output_path = t1.name
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t2:
                web_ready_output = t2.name
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as t3:
                csv_output_path = t3.name

            # 3. OpenCV Setup
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

            counted_ids = set()
            class_counts = {}
            class_names = model.names

            # 5% margin as discussed previously
            margin_x = int(width * 0.05)
            margin_y = int(height * 0.05)
            zone_x1, zone_y1 = margin_x, margin_y
            zone_x2, zone_y2 = width - margin_x, height - margin_y

            # 4. Processing Loop
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1
                if frame_count % 10 == 0: # Update progress bar every 10 frames
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                results = model.track(frame, persist=True, conf=0.35, tracker="bytetrack.yaml", verbose=False)
                annotated_frame = results[0].plot()

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().tolist()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()

                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        class_name = class_names[class_id]
                        x1, y1, x2, y2 = box
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        if zone_x1 < cx < zone_x2 and zone_y1 < cy < zone_y2:
                            if track_id not in counted_ids:
                                counted_ids.add(track_id)
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                cv2.rectangle(annotated_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 2)
                
                # Draw text background and counts
                y_offset = 50
                cv2.rectangle(annotated_frame, (20, 20), (350, 60 + max(1, len(class_counts))*40), (0, 0, 0), -1)
                cv2.putText(annotated_frame, "Total Objects:", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 40

                for cls_name, count in class_counts.items():
                    cv2.putText(annotated_frame, f"{cls_name.capitalize()}: {count}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 40

                out.write(annotated_frame)

            cap.release()
            out.release()
            progress_bar.empty()

            # 5. Save CSV
            with open(csv_output_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Object Type", "Total Count"])
                for cls_name, count in class_counts.items():
                    writer.writerow([cls_name.capitalize(), count])

            # 6. Convert Video for Web (H264 codec)
            st.text("Converting video for web playback...")
            ffmpeg_exe = imageio_ff.get_ffmpeg_exe()
            subprocess.run([ffmpeg_exe, "-y", "-i", raw_output_path, "-vcodec", "libx264", web_ready_output], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            st.success("Processing Complete!")
            st.write("### Final Counts", class_counts)

            # 7. Display Final Video and Download Buttons
            st.video(web_ready_output)
            
            col1, col2 = st.columns(2)
            with col1:
                with open(web_ready_output, "rb") as f:
                    st.download_button("📥 Download Processed Video", f, file_name="counted_output.mp4", mime="video/mp4")
            with col2:
                with open(csv_output_path, "rb") as f:
                    st.download_button("📥 Download CSV Data", f, file_name="total_counts.csv", mime="text/csv")

            # Cleanup temp files
            os.remove(video_path)
            os.remove(raw_output_path)
            os.remove(web_ready_output)
            os.remove(csv_output_path)  