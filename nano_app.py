import cv2
import os
from dotenv import load_dotenv
import gradio as gr
from collections import defaultdict
import numpy as np
import time
import base64
import requests
import json
from io import BytesIO
from PIL import Image
import ssl
import urllib3

load_dotenv()
# Disable SSL warnings and configure SSL context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# NVIDIA API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY is not set. Please put it in .env")
VILA_API_URL = "https://ai.api.nvidia.com/v1/vlm/nvidia/vila"

print("VILA model will be used via NVIDIA API for video analysis and summarization")

# ---- Helper Functions ----
def encode_frame_to_base64(frame):
    """Convert OpenCV frame to base64 string for API"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Resize for API efficiency
    pil_image = pil_image.resize((512, 384))  # Larger size for better scene understanding
    
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def analyze_video_with_vila(key_frames, video_duration):
    """Use VILA to analyze and summarize the entire video"""
    try:
        if len(key_frames) < 3:
            return "Insufficient frames for analysis"
        
        # Encode key frames to base64
        encoded_frames = [encode_frame_to_base64(frame) for frame in key_frames]
        
        # Create comprehensive prompt for video analysis
        prompt = f"""Analyze this sequence of {len(key_frames)} frames from a {video_duration:.1f}-second video. Provide a detailed summary of:

1. What activities and actions are happening in the video?
2. How many people are visible and what are they doing?
3. What is the setting/environment?
4. Are there any notable movements, interactions, or changes over time?
5. Overall description of the scene and story.

Please write a natural, flowing description as if you're describing the video to someone who can't see it."""

        # Prepare the request payload
        payload = {
            "model": "nvidia/vila",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {"url": frame}
                        } for frame in encoded_frames
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Create a session with SSL verification disabled
        session = requests.Session()
        session.verify = False
        
        print("Sending frames to VILA for comprehensive video analysis...")
        
        # Make API request with extended timeout
        response = session.post(VILA_API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            summary = result['choices'][0]['message']['content'].strip()
            return summary
        else:
            print(f"VILA API Error: {response.status_code} - {response.text}")
            return f"API Error ({response.status_code}): Could not analyze video with VILA"
            
    except requests.exceptions.SSLError as ssl_err:
        print(f"SSL Error with VILA API: {ssl_err}")
        return "SSL Error: Could not connect to VILA API"
    except requests.exceptions.RequestException as req_err:
        print(f"Request Error with VILA API: {req_err}")
        return "Network Error: Could not reach VILA API"
    except Exception as e:
        print(f"Error in VILA video analysis: {e}")
        return f"Analysis Error: {str(e)}"

def extract_key_frames(cap, total_frames, num_frames=12):
    """Extract key frames evenly distributed throughout the video"""
    key_frames = []
    
    if total_frames <= num_frames:
        # If video is short, take every frame
        frame_indices = list(range(total_frames))
    else:
        # Extract frames evenly distributed
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            key_frames.append(frame)
    
    return key_frames

# ---- Video Processing ----
def process_video(input_path):
    if input_path is None:
        return None, "Please upload a video file"
    
    try:
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            return None, "Error: Could not open video file"
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(3)), int(cap.get(4))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"Video info: {duration:.1f}s, {total_frames} frames, {fps} FPS, {w}x{h}")

        # Extract key frames for VILA analysis
        print("Extracting key frames for VILA analysis...")
        key_frames = extract_key_frames(cap, total_frames, num_frames=15)
        print(f"Extracted {len(key_frames)} key frames")

        # Reset video capture for output generation
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Create output video (optional - can be original or with overlay)
        output_path = "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_count = 0
        start_time = time.time()

        # Copy original video to output (you can add overlays here if needed)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Optional: Add timestamp or frame number overlay
            timestamp = frame_count / fps
            cv2.putText(frame, f"Time: {timestamp:.1f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)
            
            # Progress indicator
            if frame_count % (fps * 2) == 0:  # Every 2 seconds
                progress = frame_count / total_frames * 100
                print(f"Video processing progress: {progress:.1f}%")

        cap.release()
        out.release()

        # Analyze video with VILA
        print("Analyzing video content with VILA...")
        vila_summary = analyze_video_with_vila(key_frames, duration)

        # ---- Build Complete Summary ----
        summary = "🎥 VIDEO ANALYSIS REPORT\n"
        summary += "=" * 50 + "\n\n"
        summary += f"📊 Technical Details:\n"
        summary += f"• Duration: {duration:.2f} seconds\n"
        summary += f"• Total Frames: {total_frames}\n"
        summary += f"• Frame Rate: {fps} FPS\n"
        summary += f"• Resolution: {w}x{h}\n"
        summary += f"• Processing Time: {time.time() - start_time:.1f} seconds\n\n"
        
        summary += f"🤖 VILA AI Analysis:\n"
        summary += "-" * 30 + "\n"
        summary += vila_summary + "\n\n"
        
        summary += f"📝 Analysis Method:\n"
        summary += f"• Analyzed {len(key_frames)} key frames using VILA\n"
        summary += f"• AI Model: NVIDIA VILA (Vision-Language Assistant)\n"
        summary += f"• Frame sampling: Evenly distributed across video duration\n"

        print("Video analysis complete!")
        return output_path, summary
        
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        print(error_msg)
        return None, error_msg

# ---- Gradio UI ----
def create_interface():
    with gr.Blocks(title="VILA Video Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎥 VILA Video Analysis & Summarization")
        gr.Markdown("Upload a video and let VILA AI describe what's happening in natural language!")
        gr.Markdown("✨ **Powered by NVIDIA VILA**: Advanced AI that understands video content and provides human-like descriptions")

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Video(label="📁 Upload Video", sources=["upload"])
                run_btn = gr.Button("🚀 Analyze Video with VILA", variant="primary", size="lg")
                
                with gr.Accordion("ℹ️ How it works", open=False):
                    gr.Markdown("""
                    **VILA Video Analysis Process:**
                    
                    1. **Frame Extraction**: Captures key frames throughout your video
                    2. **AI Analysis**: VILA examines these frames to understand the scene
                    3. **Natural Description**: Generates human-like description of activities
                    4. **Comprehensive Summary**: Provides context, actions, and story
                    
                    **What VILA can describe:**
                    - People and their activities (walking, running, talking, etc.)
                    - Environmental context (indoor/outdoor, setting, objects)
                    - Interactions between people
                    - Changes and progression over time
                    - Overall mood and atmosphere of the scene
                    """)
            
            with gr.Column(scale=2):
                out_vid = gr.Video(label="📹 Processed Video")
                out_txt = gr.Textbox(
                    label="🤖 VILA Analysis Report", 
                    lines=20, 
                    max_lines=30,
                    show_copy_button=True
                )

        # Add examples if needed
        gr.Markdown("### 💡 Tips for best results:")
        gr.Markdown("• Videos with clear subjects work best • 15-60 seconds ideal length • Good lighting helps accuracy")

        run_btn.click(
            fn=process_video, 
            inputs=inp, 
            outputs=[out_vid, out_txt],
            show_progress=True
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        debug=True
    )