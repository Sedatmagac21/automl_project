
# 🤖 AutoML Streamlit Uygulaması

![AutoML Uygulaması Görseli](https://i.imgur.com/example-of-automl-gui.png)

### 📌 Giriş ve Proje Hakkında

Geleneksel makine öğrenmesi (ML) süreçleri; veri ön işleme, model seçimi, hiperparametre optimizasyonu ve sonuçların değerlendirilmesi gibi karmaşık ve zaman alıcı adımlar gerektirir. Bu durum, ML yeteneklerinin sadece uzman veri bilimcilerle sınırlı kalmasına yol açmaktadır.

Bu proje, **otomatik makine öğrenmesi (AutoML)** kavramını, kullanıcı dostu bir arayüzle birleştirerek bu engeli ortadan kaldırmayı hedeflemektedir. **Streamlit** framework'ü üzerinde geliştirilen bu uygulama, herhangi bir kod bilgisi gerektirmeden, kullanıcıların yalnızca veri setlerini (CSV veya Görüntü Klasörleri) yükleyerek uçtan uca ML ve Derin Öğrenme (DL) modellerini eğitmelerine olanak tanır.

Uygulama, yüklenen verinin türünü ve makine öğrenmesi görevini (Sınıflandırma, Regresyon, Görüntü Tanıma vb.) **otomatik olarak tespit eder**. Ardından, özelleştirilmiş ve optimize edilmiş ML/DL ardışık düzenlerini (pipeline) devreye sokarak, saniyeler içinde model performanslarını karşılaştırmalı olarak sunar.

Bu uygulama, hem veri bilimini öğrenenler hem de iş kararlarını hızla veri odaklı hale getirmek isteyen analistler için güçlü ve erişilebilir bir araçtır. **AutoML Streamlit Uygulaması**, karmaşık altyapıyı soyutlayarak, kullanıcıların zamanlarını model geliştirmeye değil, elde edilen içgörüleri yorumlamaya ayırmasını sağlar.

---
### ✨ Temel Özellikler

* **Veri Tipi Desteği:** Hem tablosal verileri (CSV) hem de görüntü verilerini işleme yeteneği.
* **Otomatik Görev Tespiti:** Yüklenen veriye dayanarak makine öğrenmesi görev tipini (Classification, Regression vb.) otomatik olarak belirleme.
* **Kullanıcı Arayüzü:** Streamlit framework'ü sayesinde hızlı ve interaktif bir web arayüzü sunar.
* **Modüler Mimari:** Tablosal veriler ve görüntüler için ayrı ayrı optimize edilmiş işlem hatları (`pipelines/tablolar_pipeline.py`, `pipelines/image_pipeline.py`) kullanır.
* **Geniş ML/DL Desteği:** Scikit-learn ve TensorFlow/Keras gibi popüler kütüphaneleri kullanarak hem geleneksel ML hem de derin öğrenme (DL) modellerini uygulayabilir.

### 🛠️ Kullanılan Teknolojiler

Proje, ağırlıklı olarak aşağıdaki Python kütüphaneleri üzerine kurulmuştur:

* **Arayüz:** `streamlit`
* **Veri Manipülasyonu:** `pandas`, `numpy`
* **Makine Öğrenmesi:** `scikit-learn`
* **Derin Öğrenme:** `tensorflow`
* **Görüntü İşleme:** `opencv-python`
* **Görselleştirme:** `matplotlib`, `seaborn`

### 🚀 Projeyi Adım Adım Çalıştırma

Projenin yerel makinenizde çalıştırılması için aşağıdaki adımları takip edin:

#### 1. Sanal Ortam Oluşturma ve Aktifleştirme

Python çakışmalarını önlemek için bir sanal ortam oluşturun ve etkinleştirin:

```bash
# Sanal ortam oluşturma
python -m venv env

# Sanal ortamı aktif hale getirme (Windows için)
env\Scripts\activate

# Sanal ortamı aktif hale getirme (Linux/macOS için)
source env/bin/activate
```
### Bağımlılıkları Yükleme

pip install -r requirements.txt

### Uygulamayı Başlatma

streamlit run main.py

### Nasıl Kullanılır?

1. Uygulama arayüzünde "Tabular Veri (CSV)" veya "Görüntü Klasörü" seçeneklerinden birini seçin.

2. CSV seçeneği için dosyanızı yükleyin. Görüntü klasörü için ise klasörün tam yolunu girin.

3. Uygulama, veriyi önizler, görevi tespit eder ve otomatik ML sürecini başlatır.
