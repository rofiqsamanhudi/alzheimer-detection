# Alzheimer Disease Detection from Brain MRI
Sebuah pipeline end-to-end untuk mendeteksi penyakit Alzheimer dari citra otak, mengintegrasikan pendekatan **Machine Learning klasik**, **Convolutional Neural Networks (CNN)**, dan **Vision Transformers**. Pipeline ini membandingkan performa berbagai metode, termasuk **XGBoost dengan fitur klasik** (FFT, GIST, GLCM, HOG, Hu moments, LBP, Wavelet, Zernike), **model CNN** (CNN Scratch, Resnet50, EfficientNet), dan **model Transformer** (Swin Transformer, EfficientFormer, ViT B/16). Selain itu, pipeline mendukung eksperimen dengan strategi **fine-tuning** dan **LoRA (Low-Rank Adaptation)** untuk meningkatkan akurasi deteksi pada dataset citra medis Alzheimer.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Running the Pipeline](#running-the-pipeline)
- [Evaluation Results](#evaluation-results)
- [License & Credits](#license--credits)

---

## Overview

Proyek ini mengimplementasikan sistem deteksi penyakit Alzheimer berbasis citra MRI otak menggunakan tiga pendekatan utama: Machine Learning klasik, Convolutional Neural Networks (CNN), dan Vision Transformers (ViT). Seluruh pipeline — mulai dari preprocessing, ekstraksi fitur, pelatihan model, evaluasi, hingga perbandingan performa — telah terintegrasi dalam satu notebook dan skrip Python.

### 1. Machine Learning Klasik
Pendekatan berbasis ekstraksi fitur hand-crafted + classifier tree-boosting.

**Fitur yang diekstrak:**
- FFT (Fast Fourier Transform) — menggunakan `scipy.fft.fft2`
- GLCM (Gray-Level Co-occurrence Matrix) — via `mahotas` atau `skimage`
- HOG (Histogram of Oriented Gradients) — `skimage.feature.hog`
- Hu Moments — via `cv2.HuMoments`
- LBP (Local Binary Patterns) — uniform, `skimage.feature.local_binary_pattern`
- Wavelet Transform — `pywt` (db4, 3 level assumed)
- Zernike Moments — `mahotas.features.zernike_moments` (order 10 assumed)
- GIST descriptor — implementasi custom

**Classifier:**  
XGBoost (`XGBClassifier`, default parameters atau tuned via GridSearchCV; contoh: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, objective='multi:softmax' dengan 4 kelas)

### 2. Convolutional Neural Networks (CNNs)
Transfer learning dari model pre-trained ImageNet (input 224×224×3).

| Model                | Pre-trained Weights                  | Mode Training yang Diuji                          |
|----------------------|--------------------------------------|----------------------------------------------------|
| CNN dari Scratch     | Tidak ada (from scratch)             | Baseline (tanpa pre-training)                     |
| ResNet50             | `ResNet50_Weights.IMAGENET1K_V1`     | Baseline (freeze) • Fine-tune normal • LoRA (r=8) |
| EfficientNetB0       | `EfficientNet_B0_Weights.IMAGENET1K_V1` | Baseline (freeze) • Fine-tune normal • LoRA (r=16) |

**Parameter training umum:**  
- Optimizer: Adam / AdamW (assumed)  
- Loss: Categorical Crossentropy  
- Batch size: 16–32 (assumed)  
- Epochs: hingga 50 (dengan EarlyStopping)  
- Learning rate: 0.0001–0.00001 (assumed, dengan scheduler seperti ReduceLROnPlateau)  
- Augmentasi: rotasi ±10°, horizontal flip, zoom 0.1 (via `torchvision.transforms` atau `ImageDataGenerator`)  

### 3. Vision Transformers
Menggunakan model dari `torchvision` dan `timm` library + LoRA via `loralib`.

| Model                | Pre-trained Checkpoint / Source       | Mode Training yang Diuji                          |
|----------------------|--------------------------------------|----------------------------------------------------|
| Swin Transformer     | `swin_t` dari `torchvision.models`   | Baseline • Fine-tune normal • LoRA (r=8)          |
| EfficientFormer      | `timm.create_model` (e.g., L1)       | Baseline • Fine-tune normal • LoRA (r=4)          |
| ViT Base /16         | `timm.create_model` (e.g., `vit_base_patch16_224`) | Baseline • Fine-tune normal • LoRA (r=16, target query/value) |

**Parameter training umum ViT:**  
- Optimizer: AdamW  
- Learning rate: 5e-5 → 1e-5 (fine-tune)  
- Batch size efektif: 32 (dengan gradient accumulation)  
- Epochs: hingga 30 (dengan EarlyStopping)  
- Scheduler: CosineAnnealing + warmup  

### Dataset
Dataset terdiri dari citra MRI otak dari sumber seperti Kaggle (e.g., Alzheimer's Dataset) atau ADNI, dengan total sekitar 7,811 sampel. Distribusi kelas (4-class) sebagai berikut (berdasarkan contoh di notebook):

| Kelas                  | Jumlah Sampel Aproksimasi | Class Weight (Balanced) |
|------------------------|---------------------------|-------------------------|
| No Impairment (Non-Demented) | ~3,500                  | 0.882                   |
| Very Mild Impairment   | ~2,000                    | 0.969                   |
| Mild Impairment        | ~1,200                    | 1.079                   |
| Moderate Impairment    | ~1,100                    | 1.101                   |

- **Split:** 60% train (~4,687), 20% validation (~1,562), 20% test (~1,562) dengan stratifikasi.
- **Sumber:** Download dari [Kaggle Alzheimer's Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images) atau ADNI. Simpan di folder `dataset/` dengan sub-folder per kelas (e.g., `dataset/train/No Impairment/`).
- **Catatan:** Dataset mentah disimpan di `dataset/`, hasil preprocessing di `processed_dataset/`. Pastikan citra dalam format JPEG/PNG, grayscale atau RGB.

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
   - Model CNN: CNN Scratch, Resnet50, EfficientNet.
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
| Resnet50     | 72.67%   | 78.40%             | **95.45%**       |
| EfficientNet | 94.98%   | 93.09%             | 94.27%           |

> **Catatan:** Resnet50 dengan **LoRA Fine-Tuning** mencapai akurasi tertinggi 95.45%. Arsitektur residual Resnet50 membantu stabilitas jaringan dalam, dan LoRA memungkinkan penyesuaian bobot secara efisien tanpa overfitting, sehingga proses pelatihan lebih efektif.

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
   Secara keseluruhan, hasil eksperimen menunjukkan bahwa setiap pendekatan memiliki karakteristik dan keunggulan yang berbeda. Metode berbasis fitur klasik cenderung memberikan performa yang baik pada dataset berukuran relatif kecil, karena proses ekstraksi fitur manual sudah mampu menangkap pola visual dasar. Sementara itu, arsitektur CNN menunjukkan kemampuan yang lebih baik dalam mempelajari pola lokal dan karakteristik tekstur yang lebih kompleks pada citra medis.

   Pendekatan berbasis Transformer memberikan hasil tertinggi karena mekanismenya yang mampu memodelkan hubungan spasial global antar patch citra secara lebih efektif. Selain itu, eksperimen juga menegaskan pentingnya melakukan perbandingan yang komprehensif antar berbagai pendekatan, meliputi metode klasik, deep learning konvensional, transfer learning, serta teknik fine-tuning berbasis LoRA. Perbandingan ini diperlukan untuk mengevaluasi tidak hanya performa akurasi, tetapi juga efisiensi komputasi, kebutuhan data, dan tingkat generalisasi masing-masing metode.

---

## License & Credits

Proyek ini dikembangkan untuk tujuan edukasi dan eksperimen **Computer Vision – Deteksi Alzheimer**.

**Kontributor:**
- Rofiq Samanhudi
- Muhammad Ikbar Ananda Sulistio
- Dimas Arief Wicaksono

**License**

Repository ini dilisensikan di bawah [MIT License](LICENSE), yang memungkinkan penggunaan pribadi, edukatif, dan redistribusi dengan menyertakan atribusi yang sesuai.
