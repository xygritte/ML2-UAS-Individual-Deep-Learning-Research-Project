# ML2 UAS — Individual Deep Learning Research Project

Project ini disusun mengikuti ketentuan Final Project Pembelajaran Mesin 2: minimal 1 baseline, minimal 1 model utama Deep Learning, dan minimal 3 skenario eksperimen.

## Rancangan Penelitian Sementara

**Bidang:** Computer Vision  
**Dataset:** CIFAR-10  
**Baseline:** CNN sederhana  
**Model utama:** ResNet-18 pretrained (transfer learning)  

### Skenario

| Eksperimen | Model | Konfigurasi utama |
|---|---|---|
| E1 | CNN Baseline | Dropout 0.0, Adam, LR 1e-3 |
| E2 | CNN + Dropout | Dropout 0.5, Adam, LR 1e-3 |
| E3 | ResNet-18 Transfer Learning | Pretrained, Adam, LR 1e-4 |

**Catatan:** angka hasil belum dianggap hasil penelitian sampai training benar-benar dijalankan. Program sengaja menghasilkan `history`, metrik test, prediksi, dan confusion matrix agar analisis overfitting/underfitting serta error analysis dapat dilakukan.

## Struktur

```text
.
├── data/                    # Dataset CIFAR-10 (diunduh otomatis)
├── outputs/                 # Hasil training dan evaluasi
├── src/
│   ├── dataset.py           # Dataset dan preprocessing
│   ├── models.py            # Baseline CNN dan ResNet-18
│   ├── train.py             # Training E1/E2/E3
│   └── evaluate.py          # Metrik, confusion matrix, kurva training
├── requirements.txt
└── README.md
```

## Menjalankan Program

### 1. Buat virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Jika PowerShell memblokir `Activate.ps1`, gunakan Python dari `.venv` langsung atau izinkan script hanya untuk sesi tersebut:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependensi

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Uji dataset dan baseline

```powershell
python src/train.py --experiment E1 --epochs 1 --batch-size 128
```

Perintah ini digunakan sebagai **smoke test**. CIFAR-10 akan diunduh otomatis pada eksekusi pertama.

### 4. Jalankan eksperimen

Untuk pengujian awal yang relatif ringan:

```powershell
python src/train.py --experiment all --epochs 3 --batch-size 128
```

Untuk eksperimen penelitian yang lebih lengkap, jumlah epoch dapat dinaikkan setelah smoke test berhasil, misalnya:

```powershell
python src/train.py --experiment all --epochs 20 --batch-size 128
```

### 5. Buat hasil evaluasi

```powershell
python src/evaluate.py
```

Hasil akan tersimpan di `outputs/`, termasuk:

- `*_history.csv`
- `*_metrics.json`
- `*_predictions.csv`
- `*_training_curves.png`
- `*_confusion_matrix.png`
- `summary_metrics.csv`

## Prinsip Validitas Eksperimen

Semua accuracy, precision, recall, F1-score, loss, grafik, dan kesimpulan harus berasal dari training yang benar-benar dijalankan. Nilai pada paper tidak boleh dibuat secara manual.
