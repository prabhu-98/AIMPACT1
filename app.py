# app.py
import streamlit as st
import time
from moviepy.video.io.VideoFileClip import VideoFileClip
import speech_recognition as sr
import google.generativeai as genai
import cv2
from PIL import Image
import tempfile
import os

# Configure Gemini API key
GOOGLE_API_KEY = 'AIzaSyDQRPWOtlzJZFWwGiU7j3fg9_gUdsPNFsU'
genai.configure(api_key=GOOGLE_API_KEY)

def extract_frames(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_indices = [total_frames * i // num_frames for i in range(num_frames)]
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    return frames

def analyze_speech(video_file):
    try:
        with st.spinner("Processing audio..."):
            # Create temporary file for video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video.write(video_file.read())
                video_path = temp_video.name

            # Extract audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(temp_audio_path, logger=None)
                video.close()

        with st.spinner("Transcribing speech..."):
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio = recognizer.record(source)
            transcription = recognizer.recognize_google(audio)

        st.subheader("Speech Transcription")
        st.write(transcription)
        
        with st.spinner("Analyzing speech content..."):
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
            
            speech_prompt = f"""
            Give a precise answer in under 200 words
            Analyze the following speech transcript:
                - **Key Topics & Themes**
                - **Main Arguments & Messages**
                - **Clarity, Tone & Language**
                - **Strengths & Areas for Improvement**
                - **Actionable Suggestions**
            Transcript:
            {transcription}
          """
            
        speech_analysis = model.generate_content(speech_prompt).text
            
        st.subheader("Speech Analysis")
        st.markdown(speech_analysis)
        
        # Clean up
        os.remove(temp_video.name)
        os.remove(temp_audio_path)
        
        st.success("Speech Analysis Complete!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

def analyze_body_language(video_file):
    try:
        with st.spinner("Processing video..."):
            # Create temporary file for video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video.write(video_file.read())
                video_path = temp_video.name

            frames = extract_frames(video_path)
            
        st.subheader("Extracted Frames")
        cols = st.columns(len(frames))
        for idx, (frame, col) in enumerate(zip(frames, cols)):
            col.image(frame, caption=f"Frame {idx+1}", use_container_width=True)
        
        with st.spinner("Analyzing body language..."):
            model_vision = genai.GenerativeModel('gemini-1.5-pro-latest')
            
            frame_analyses = []
            progress_bar = st.progress(0)
            
            for i, frame in enumerate(frames):
                prompt = """
                             Give a precise answer in under 100 words
                            Analyze this frame for body language:
                            - Facial expressions & emotions
                            - Eye contact & gaze direction
                            - Body posture & hand gestures
                            - Confidence level & improvements
                            Provide a structured and concise response.
                 """
                response = model_vision.generate_content([prompt, frame])
                frame_analyses.append(f"Frame {i+1} Analysis:\n{response.text}")
                progress_bar.progress((i + 1) / len(frames))
                time.sleep(2)  # Add delay between API calls
            
        st.subheader("Body Language Analysis")
        for analysis in frame_analyses:
            st.markdown("---")
            st.markdown(analysis)
            
        # Clean up
        os.remove(temp_video.name)
        
        st.success("Body Language Analysis Complete!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    st.set_page_config(page_title="Video Analysis Tool", page_icon="🎥", layout="wide")
    
    st.title("🎥 Video Analysis Tool")
    st.write("Upload a video to analyze speech and body language.")
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        analysis_type = st.radio(
            "Choose analysis type:",
            ["Speech Analysis", "Body Language Analysis", "Both"],
            horizontal=True
        )
        
        if st.button("Start Analysis"):
            if analysis_type in ["Speech Analysis", "Both"]:
                st.header("🎤 Speech Analysis")
                analyze_speech(uploaded_file)
                
            if analysis_type in ["Body Language Analysis", "Both"]:
                st.header("👁️ Body Language Analysis")
                uploaded_file.seek(0)  # Reset file pointer
                analyze_body_language(uploaded_file)

if __name__ == "__main__":
    main()
