# Alzheimer Detection
Sebuah pipeline end-to-end untuk mendeteksi penyakit Alzheimer dari citra otak, mengintegrasikan pendekatan **Machine Learning klasik**, **Convolutional Neural Networks (CNN)**, dan **Vision Transformers**. Pipeline ini membandingkan performa berbagai metode, termasuk **XGBoost dengan fitur klasik** (FFT, GIST, GLCM, HOG, Hu moments, LBP, Wavelet, Zernike), **model CNN** (CNN Scratch, ResNet50, EfficientNet), dan **model Transformer** (Swin Transformer, EfficientFormer, ViT B/16). Selain itu, pipeline mendukung eksperimen dengan strategi **fine-tuning** dan **LoRA (Low-Rank Adaptation)** untuk meningkatkan akurasi deteksi pada dataset citra medis Alzheimer.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Running the Pipeline](#running-the-pipeline)
- [Evaluation Results](#evaluation-results)
- [License & Credits](#license--credits)

---

## Overview

Proyek ini mengimplementasikan sistem deteksi Alzheimer dari citra otak dengan beberapa pendekatan:

1. **Machine Learning Klasik**  
   Menggunakan fitur citra yang diekstraksi:
   - FFT, GLCM, HOG, Hu Moments, LBP, Wavelet, Zernike, GIST
   - Classifier: XGBoost

2. **Convolutional Neural Networks (CNNs)**  
   Model yang digunakan:
   - CNN Scratch
   - ResNet50
   - EfficientNet  
   Mode training: baseline, fine-tune normal, dan LoRA

3. **Vision Transformers**  
   Model yang digunakan:
   - Swin Transformer
   - EfficientFormer
   - ViT B/16  
   Mode training: baseline, fine-tune normal, dan LoRA

Pipeline mencakup **preprocessing data, ekstraksi fitur, pelatihan model, evaluasi, dan perbandingan hasil** — semua terintegrasi dalam skrip dan notebook di repositori.

---

## Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/rofiqsamanhudi/alzheimer-detection.git
cd alzheimer-detection
```

### 2. Buat virtual environment
```bash
python -m venv venv

# Windows (CMD)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Jika terjadi error policy di PowerShell, jalankan:
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Jika `requirements.txt` masih kosong, Anda dapat membuatnya secara otomatis:
```bash
pip install pipreqs
pipreqs . --force
```


---

## Project structure

```
alzheimer-detection/
├── dataset/                     # Dataset mentah citra medis
├── processed_dataset/           # Dataset hasil preprocessing & fitur klasik
├── Fitur_Ekstraksi_Klasik/      # Folder untuk preprocessing & ekstraksi fitur ML klasik
├── CNN_models/                  # Model CNN scratch dan hasil training
├── ResNet_models/               # Model ResNet50 dan hasil training
├── EfficientNet_models/         # Model EfficientNet dan hasil training
├── EfficientFormer/             # Model EfficientFormer Transformer
├── Swin_Transformer/            # Model Swin Transformer
├── ViT-B16_models/              # Model Vision Transformer B/16
├── alzheimer_detection.ipynb    # Notebook utama dan eksperimen
├── Akademik poster              # Poster akademik
├── Alzheimer-Detection.pdf      # Laporan Hasil Percobaan
├── requirements.txt             # Dependencies Python
├── README.dataset.txt           # Informasi dataset
├── README.roboflow.txt          # Catatan penggunaan Roboflow
├── LICENSE                      # Lisensi proyek
└── README.md                    # Dokumentasi utama proyek
```

> **Catatan:** Struktur ini memudahkan pengelolaan dataset, fitur, model, dan eksperimen, sehingga seluruh pipeline end-to-end Alzheimer Detection bisa dijalankan secara terorganisir.


---
## Running the pipeline

Semua tahapan dijalankan melalui notebook: `alzheimer_detection.ipynb`.

### ❐ Tahapan Pipeline:

1. **Preprocessing & Ekstraksi Fitur**
   - Normalisasi dan resize citra.
   - Ekstraksi fitur klasik: FFT, GIST, GLCM, HOG, Hu, LBP, Wavelet, Zernike.
   - Simpan fitur di `processed_dataset/`

2. **Training Machine Learning Klasik**
   - Latih model XGBoost menggunakan fitur klasik.
   - Evaluasi model pada validation/test set.

3. **Training Deep Learning**
   - Model CNN: CNN Scratch, ResNet50, EfficientNet.
   - Model Transformer: Swin, EfficientFormer, ViT-B/16.
   - Evaluasi model pada validation/test set.
   - Latih dengan strategi: baseline, fine-tuning normal, dan LoRA.

4. **Evaluasi**
   - Hitung Akurasi, Confusion Matrix, dan Classification Report.
   - Simpan hasil pada setiap folder model.

### ❐ Jalankan Notebook
```bash
jupyter notebook alzheimer_detection.ipynb
```

---
## Evaluation results

### 1️. Machine Learning Klasik (XGBoost)

| Fitur   | Akurasi Validation |
| :-----: | :---------------: |
| FFT     | 78.19%            |
| GIST    | **92.93%**        |
| GLCM    | 86.38%            |
| HOG     | 88.79%            |
| Hu      | 88.58%            |
| LBP     | 68.97%            |
| Wavelet | 76.70%            |
| Zernike | 80.34%            |

> **Catatan:** Fitur **GIST** memberikan performa terbaik di antara fitur klasik lainnya. GIST mampu menangkap informasi global dan tekstur citra dengan baik, sehingga XGBoost dapat memanfaatkan pola yang lebih representatif.

---

### 2️. Convolutional Neural Networks (CNN)

| Model        | Baseline | Fine-Tuning Normal | LoRA Fine-Tuning |
| :----------: | :------: | :----------------: | :--------------: |
| CNN Scratch  | 95.29%   | -                  | -                |
| ResNet50     | 72.67%   | 78.40%             | **95.45%**       |
| EfficientNet | 94.98%   | 93.09%             | 94.27%           |

> **Catatan:** ResNet50 dengan **LoRA Fine-Tuning** mencapai akurasi tertinggi 95.45%. Arsitektur residual ResNet50 membantu stabilitas jaringan dalam, dan LoRA memungkinkan penyesuaian bobot secara efisien tanpa overfitting, sehingga proses pelatihan lebih efektif.

---

### 3️. Transformers

| Model            | Baseline | Fine-Tuning Normal | LoRA Fine-Tuning |
| :--------------: | :------: | :----------------: | :--------------: |
| Swin Transformer | 93.71%   | 95.34%             | 94.73%           |
| EfficientFormer  | 95.14%   | 93.14%             | 94.73%           |
| ViT B/16         | **96.26%** | **96.57%**       | 92.78%           |

> **Catatan:** **ViT B/16** dengan Fine-Tuning Normal menghasilkan akurasi tertinggi 96.57%. Vision Transformer unggul karena mampu menangkap hubungan spasial antar patch citra secara global, dan Fine-Tuning Normal menyesuaikan representasi fitur dengan dataset Alzheimer, sehingga prediksi lebih akurat.

---

**Catatan Umum:**  
- Fitur klasik unggul pada dataset kecil dan sederhana karena ekstraksi fitur manual sudah cukup menangkap pola utama.  
- CNN lebih baik untuk menangkap pola lokal yang kompleks dalam citra medis.  
- Transformers unggul pada hubungan spasial global dan fitur kompleks, sehingga akurasi tertinggi dicapai di kategori ini.

---

## License & Credits

Proyek ini dikembangkan untuk tujuan edukasi dan eksperimen **Computer Vision – Deteksi Alzheimer**.

**Kontributor:**
- Rofiq Samanhudi
- Muhammad Ikbar Ananda Sulistio
- Dimas Arief Wicaksono

**License**

Repository ini dilisensikan di bawah [MIT License](LICENSE), yang memungkinkan penggunaan pribadi, edukatif, dan redistribusi dengan menyertakan atribusi yang sesuai.
