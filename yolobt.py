import supervision as sv
from ultralytics import YOLO 
from tqdm import tqdm
import argparse
import numpy as np
import streamlit as st
import json
import os
from glob import glob

tracker = sv.ByteTrack() 

class_counts_ui = st.empty()
class_counts = {'car': {'count': 0, 'avg_duration': 0, 'total_duration': 0},
                'truck': {'count': 0, 'avg_duration': 0, 'total_duration': 0}}

frame_placeholder = st.empty()

progress_bar = st.progress(0)

fps = 30.0  # Video FPS değeri, videoya özel olarak ayarlanmalı
frame_duration = 1 / fps

def process_video(
        source_weights_path: str, 
        source_video_path: str,
        target_video_path: str, 
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7
) -> None:
    model = YOLO(source_weights_path)       # Load YOLO model 
    classes = list(model.names.values())    # Class names 
    LINE_STARTS = sv.Point(0,500)           # Line start point for count in/out vehicle
    LINE_END = sv.Point(1280, 500)          # Line end point for count in/out vehicle
    tracker = sv.ByteTrack()                # Bytetracker instance 
    box_annotator = sv.BoundingBoxAnnotator()     # BondingBox annotator instance 
    label_annotator = sv.LabelAnnotator()         # Label annotator instance 
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path) # for generating frames from video
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    line_counter = sv.LineZone(start=LINE_STARTS, end = LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale= 0.5)
    
    unique_trackers = {'car': set(), 'truck': set()}

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        # total frames for progress bar
        total_frames = video_info.total_frames
        
        for frame_count, frame in enumerate(tqdm(frame_generator, total=total_frames)):
            # Getting result from model
            results = model(frame, verbose=False, conf= confidence_threshold, iou = iou_threshold)[0] 
            detections = sv.Detections.from_ultralytics(results)    # Getting detections
            #Filtering classes for car and truck only instead of all COCO classes.
            detections = detections[np.where((detections.class_id==2)|(detections.class_id==7))]
            detections = tracker.update_with_detections(detections)  # Updating detection to Bytetracker
            # Annotating detection boxes
            annotated_frame = box_annotator.annotate(scene = frame.copy(), detections= detections) 
            
            progress_bar.progress((frame_count + 1) / total_frames)
            
            # Prepare labels and count classes
            labels = []
            for index in range(len(detections.class_id)):
                class_name = classes[detections.class_id[index]]
                tracker_id = detections.tracker_id[index]
                confidence = round(detections.confidence[index], 2)
                label = f"#{tracker_id} {class_name} {confidence}"
                labels.append(label)
                if tracker_id not in unique_trackers[class_name]:
                    unique_trackers[class_name].add(tracker_id)
                    class_counts[class_name]['count'] += 1
                class_counts[class_name]['total_duration'] += frame_duration

            # Update average durations
            for key in class_counts:
                if class_counts[key]['count'] > 0:
                    class_counts[key]['avg_duration'] = (class_counts[key]['total_duration'] / class_counts[key]['count']) / 60
            
            with class_counts_ui.container():
                st.write(class_counts)
            
            annotated_label_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            frame_placeholder.image(annotated_label_frame, channels="BGR", use_column_width=True)
            
            sink.write_frame(frame=annotated_label_frame)
            
            # Line counter in/out trigger
            line_counter.trigger(detections=detections)
            # Annotating labels
            annotated_label_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            # Annotating line labels
            line_annotate_frame = line_annotator.annotate(frame=annotated_label_frame, line_counter=line_counter)
            sink.write_frame(frame = line_annotate_frame)
        
        #class_counts öğesini json olarak kaydetme
        # Hedef klasör yolu
        target_folder = 'outputs/jsons'

        # Eğer hedef klasör yoksa, oluştur
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Mevcut dosyaları sayarak yeni dosya numarasını belirleme
        existing_files = glob(os.path.join(target_folder, 'class_counts_run*.json'))
        next_file_number = len(existing_files) + 1  # Mevcut dosya sayısına 1 ekleyerek yeni dosya numarası

        # Yeni dosya adı
        file_name = f"class_counts_run{next_file_number}.json"
        file_path = os.path.join(target_folder, file_name)

        # JSON dosyasına kaydetme
        with open(file_path, 'w') as json_file:
            json.dump(class_counts, json_file, indent=4)

        print(f'Class counts saved as JSON in {file_path}.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("video processing with YOLO and ByteTrack") 
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str
    )
    parser.add_argument(
        "--source_video_path",
        required=True, 
        help="Path to the source video file",
        type = str
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file",
        type= str
    )
    parser.add_argument(
        "--confidence_threshold",
        default = 0.3,
        help= "Confidence threshold for the model",
        type=float
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        help="Iou threshold for the model",
        type= float
    )
    args = parser.parse_args() 
    process_video(
        source_weights_path=args.source_weights_path, 
        source_video_path= args.source_video_path,
        target_video_path=args.target_video_path, 
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold
    )