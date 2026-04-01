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
@st.cache_resource
def load_model():
    return YOLO("Models/best.pt") 

model = load_model()

# --- FILE UPLOAD ---
uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'mov', 'avi'])

if uploaded_video is not None:
    
    # --- EXTRACT FIRST FRAME FOR PREVIEW (Optimized with Session State) ---
    if "video_name" not in st.session_state or st.session_state.video_name != uploaded_video.name:
        st.session_state.processing_complete = False
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_video.read())
            tmp_path = tfile.name
        
        cap = cv2.VideoCapture(tmp_path)
        success, frame = cap.read()
        st.session_state.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        st.session_state.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        st.session_state.fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if success:
            # Convert BGR to RGB for Streamlit image display
            st.session_state.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        cap.release()
        os.remove(tmp_path) # Clean up preview temp file immediately
        st.session_state.video_name = uploaded_video.name
        
        
        uploaded_video.seek(0)


    st.write("### 🎛️ Adjust Counting Zone")
    col1, col2 = st.columns(2)
    with col1:
        margin_x_pct = st.slider("Left/Right Margin (%)", min_value=0, max_value=45, value=5, step=1)
    with col2:
        margin_y_pct = st.slider("Top/Bottom Margin (%)", min_value=0, max_value=45, value=5, step=1)

    # Calculate actual pixel margins based on slider percentages
    margin_x = int(st.session_state.width * (margin_x_pct / 100.0))
    margin_y = int(st.session_state.height * (margin_y_pct / 100.0))
    
    zone_x1, zone_y1 = margin_x, margin_y
    zone_x2 = st.session_state.width - margin_x
    zone_y2 = st.session_state.height - margin_y

    # --- DISPLAY LIVE PREVIEW ---
    if 'first_frame' in st.session_state:
        preview_img = st.session_state.first_frame.copy()
        cv2.rectangle(preview_img, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 4)
        cv2.putText(preview_img, "Counting Zone Preview", (zone_x1 + 15, zone_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        st.image(preview_img, caption="Live Zone Preview (Adjust sliders above to change size)", use_container_width=True)

    # --- PROCESSING BUTTON ---
    if st.button("🚀 Start Processing Video"):
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
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(raw_output_path, fourcc, st.session_state.fps, (st.session_state.width, st.session_state.height))

            counted_ids = set()
            class_counts = {}
            class_names = model.names

            # 4. Processing Loop
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1
                if frame_count % 10 == 0: 
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

                        # Use dynamic zone coordinates from the sliders
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

    
            # Read the files into memory so they survive the app resetting
            with open(web_ready_output, "rb") as f:
                st.session_state.processed_video = f.read()
            with open(csv_output_path, "rb") as f:
                st.session_state.processed_csv = f.read()
            
            st.session_state.final_counts = class_counts
            st.session_state.processing_complete = True

            # Cleanup temp files safely now that data is in memory
            os.remove(video_path)
            os.remove(raw_output_path)
            os.remove(web_ready_output)
            os.remove(csv_output_path)

# --- DISPLAY RESULTS AND DOWNLOADS ---
if st.session_state.get('processing_complete', False):
    st.success("✅ Processing Complete!")
    st.write("### Final Counts", st.session_state.final_counts)


    st.video(st.session_state.processed_video)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Processed Video", 
            data=st.session_state.processed_video, 
            file_name="counted_output.mp4", 
            mime="video/mp4"
        )
    with col2:
        st.download_button(
            label="📥 Download CSV Data", 
            data=st.session_state.processed_csv, 
            file_name="total_counts.csv", 
            mime="text/csv"
        )