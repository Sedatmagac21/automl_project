import streamlit as st
import pandas as pd
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from utils.helper import detect_data_type, detect_task_type
from pipelines.tablolar_pipeline import process_tabular_data
from pipelines.image_pipeline import process_image_data, predict_with_model

def setup_page():
    st.set_page_config(
        page_title="AutoML Uygulaması",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 AutoML Streamlit Uygulaması")
    st.write("CSV dosyaları ve görüntüler için otomatik makine öğrenmesi")

def process_tabular_file(file):
    try:
        data = pd.read_csv(file)
        st.write("Veri Önizleme:")
        st.dataframe(data.head())
        
        st.write("Veri Özeti:")
        st.write(f"Satır Sayısı: {data.shape[0]}")
        st.write(f"Sütun Sayısı: {data.shape[1]}")
        
        st.write("Sütun Tipleri:")
        st.write(data.dtypes)
        
        task_type = detect_task_type(data)
        st.info(f"Tespit Edilen Görev Türü: {task_type}")
        
        with st.spinner('Model eğitiliyor...'):
            model, score, download_path, download_filename = process_tabular_data(data, task_type)
            
        if model and score > float('-inf'):
            st.success("Model eğitimi tamamlandı!")
            st.write(f"En İyi Model: {type(model).__name__}")
            st.write(f"Model Başarı Skoru: {score:.4f}")
            
            if download_path and download_filename:
                with open(download_path, 'rb') as f:
                    model_bytes = f.read()
                st.download_button(
                    label="En İyi Modeli İndir",
                    data=model_bytes,
                    file_name=download_filename,
                    mime="application/octet-stream"
                )
                os.remove(download_path)
        else:
            st.warning("Model eğitimi başarısız oldu.")
        
        return model, score
        
    except Exception as e:
        st.error(f"Veri işlenirken bir hata oluştu: {str(e)}")
        return None, None
    
def process_image_directory(directory_path):
    try:
        if not os.path.isdir(directory_path):
            st.error("Geçerli bir klasör yolu değil!")
            return None, None

        class_dirs = [d for d in os.listdir(directory_path) 
                     if os.path.isdir(os.path.join(directory_path, d))]
        
        if not class_dirs:
            st.error("Sınıf klasörleri bulunamadı!")
            return None, None
            
        total_images = 0
        for class_dir in class_dirs:
            class_path = os.path.join(directory_path, class_dir)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(images)
        
        if total_images == 0:
            st.error("Hiçbir sınıf klasöründe görüntü dosyası bulunamadı!")
            return None, None

        st.info(f"Toplam {len(class_dirs)} sınıf ve {total_images} görüntü bulundu.")
        
        with st.spinner('Model eğitiliyor...'):
            model, accuracy, model_path, class_indices = process_image_data(directory_path)
        
        if model and accuracy > 0:
            st.write("---")
            st.write("### Model Test Aşaması")
            test_image = st.file_uploader(
                "Test etmek için bir görüntü yükleyin",
                type=['jpg', 'jpeg', 'png']
            )
            
            if test_image:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(test_image.getvalue())
                    
                predicted_class, confidence = predict_with_model(
                    model, 
                    tmp_file.name, 
                    class_indices
                )
                
                if predicted_class:
                    st.success(f"Tahmin: {predicted_class}")
                    st.write(f"Güven Oranı: {confidence:.2%}")
                
                os.unlink(tmp_file.name)
        
        return model, accuracy
        
    except Exception as e:
        st.error(f"Görüntü klasörü işlenirken bir hata oluştu: {str(e)}")
        return None, None

def main():
    setup_page()
    
    with st.sidebar:
        st.header("Ayarlar")
        st.write("Desteklenen veri türleri:")
        st.write("- CSV (.csv)")
        st.write("- Görüntü Klasörü (içinde .jpg, .jpeg, .png dosyaları)")

    data_type_choice = st.radio(
        "İşlem türünü seçin:",
        ["Tabular Veri (CSV)", "Görüntü Klasörü"]
    )

    if data_type_choice == "Tabular Veri (CSV)":
        uploaded_file = st.file_uploader(
            "CSV dosyası yükleyin",
            type=["csv"],
            help="CSV formatında bir dosya yükleyin"
        )

        if uploaded_file is not None:
            process_tabular_file(uploaded_file)

    else:
        directory = st.text_input(
            "Görüntü klasörünün tam yolunu girin:",
            help="Örnek: C:/Users/kullanici/resimler"
        )

        if directory:
            if os.path.isdir(directory):
                if st.button("İşlemi Başlat"):
                    process_image_directory(directory)
            else:
                st.error("Geçersiz klasör yolu! Lütfen var olan bir klasör yolu girin.")

    with st.expander("📖 Nasıl Kullanılır?"):
        st.write("""
        1. Soldaki seçeneklerden veri türünü seçin
        2. CSV dosyası için: Dosya yükleme butonunu kullanın
        3. Görüntü klasörü için: Klasör yolunu girin
        4. Sistem otomatik olarak verileri işleyecek
        5. Sonuçlar ekranda gösterilecek
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Beklenmeyen bir hata oluştu. Lütfen daha sonra tekrar deneyin.")
