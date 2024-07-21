import os
import cv2
import streamlit as st

def get_model_names():
    models_dir = 'models'  # Model dosyalarının bulunduğu klasör
    model_files = os.listdir(models_dir)
    return model_files

def get_model(model_selection):
    model = os.path.join("models", model_selection)
    return model

def show_video_preview(video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Frame'i RGB'ye çevir ve göster
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, caption='Video Önizlemesi', use_column_width=True)
    cap.release()