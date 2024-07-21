import streamlit as st
import tempfile
from app_utils import get_model, get_model_names,  show_video_preview
import yolobt

output_path = "outputs"

def main():
    st.title("YOLO ByteTrack Test")
    class_counts = yolobt.class_counts
    class_counts_ui = st.empty()
    # Video yükleme
    video_file = st.file_uploader("Video yükle", type=['mp4', 'avi', 'mov'])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")  # Uygun uzantıyı ekleyin
        tfile.write(video_file.read())
        video_path = tfile.name

        # Video önizlemesi
        #show_video_preview(video_path)

        # Model seçimi
        model_option = st.selectbox("Model Seçimi", get_model_names())
        
        # Confidence Threshold Slider
        conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        # IOU Threshold Slider
        iou_threshold = st.slider("IOU Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        # Başlat butonu
        if st.button("Başlat"):
            selected_model = get_model(model_option)
            st.write("Model çalıştırılıyor...", model_option)
            # process_video fonksiyonunu burada çağır
            results = yolobt.process_video(selected_model, video_path, output_path, conf_threshold, iou_threshold)
                    
            st.write("İşlem Tamamlandı. Sonuçları 'outputs' klasöründe görebilirsiniz.")
            st.write(results)  # Sonuçları göster
    else:
        # Video yüklenene kadar tüm elemanlar pasif
        st.selectbox("Model Seçimi", [''], disabled=True)
        st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=True)
        st.slider("IOU Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=True)
        st.button("Başlat", disabled=True)

if __name__ == "__main__":
    main()
