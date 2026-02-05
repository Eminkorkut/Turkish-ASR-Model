# 🎙️ Turkish ASR - Production-Ready Conformer

Modern, high-performance Turkish Automatic Speech Recognition system based on **Conformer** architecture with state-of-the-art techniques.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

### Model Architecture
- **Conformer Encoder** with modern enhancements
- **Flash Attention** (PyTorch 2.0+ SDPA)
- **Rotary Position Embeddings (RoPE)**
- **Multi-Query Attention (MQA)** - Memory efficient
- **SwiGLU Activation** - Modern FFN
- **GroupNorm** - Batch-independent normalization

### Data Pipeline
- **torchaudio** - GPU-accelerated mel spectrograms
- **SpeedPerturbation** - 0.9x/1.0x/1.1x augmentation
- **NoisePerturbation** - SNR-based noise injection
- **SpecAugment** - Frequency/time masking
- **BucketingSampler** - Efficient length-based batching

### Decoding
- **Greedy Decoding** - Fast inference
- **Beam Search** - Higher accuracy
- **KenLM Integration** - N-gram language model
- **Flashlight Decoder** - High-performance option

### Production
- **ONNX Export** - Platform-independent deployment
- **FastAPI Server** - REST API
- **Docker** - Containerization

## 📂 Project Structure

```
Turkish-ASR-Model/
├── data/
│   ├── dataset.py        # Dataset with BucketingSampler
│   ├── preprocessing.py  # torchaudio feature extraction
│   └── tokenizer.py      # HuggingFace tokenizer
├── model/
│   ├── conformer.py      # Conformer + SwiGLU + GroupNorm
│   └── attention.py      # RoPE + MQA + Flash Attention
├── trainer/
│   └── trainer.py        # Gradient clipping/accumulation
├── utils/
│   ├── config.py         # CLI arguments
│   ├── decoding.py       # KenLM + Beam Search
│   ├── logger.py
│   └── metrics.py        # WER/CER
├── serve/
│   └── api.py            # FastAPI server
├── main.py               # Training script
├── inference.py          # Inference script
├── export_onnx.py        # ONNX export
├── Dockerfile
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Basic training
python main.py --data_path /path/to/data --n_mel_channels 80

# With augmentation
python main.py --data_path /path/to/data --augment --speed_perturb

# With gradient accumulation (effective batch = 32 * 4 = 128)
python main.py --data_path /path/to/data \
  --batch_size 32 \
  --accumulation_steps 4 \
  --gradient_clip 1.0

# Resume training
python main.py --resume
```

### Inference

```bash
# Single file
python inference.py --audio audio.wav --model runs/best_model.pt

# With beam search
python inference.py --audio audio.wav --model runs/best_model.pt --beam_search
```

### ONNX Export

```bash
python export_onnx.py --checkpoint runs/best_model.pt --output model.onnx
```

### API Server

```bash
# Local
python serve/api.py

# Docker
docker build -t turkish-asr .
docker run -p 8000:8000 -v ./runs:/app/models turkish-asr

# Test
curl -X POST http://localhost:8000/transcribe -F "file=@audio.wav"
```

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_mel_channels` | 80 | Mel filterbanks |
| `--d_model` | 256 | Model dimension |
| `--n_heads` | 4 | Attention heads |
| `--n_blocks` | 8 | Conformer blocks |
| `--gradient_clip` | 1.0 | Max gradient norm |
| `--accumulation_steps` | 1 | Gradient accumulation |
| `--augment` | False | Enable SpecAugment |
| `--speed_perturb` | False | Enable speed perturbation |

## 📊 Metrics

Training outputs:
- **Loss** - CTC loss
- **WER** - Word Error Rate
- **CER** - Character Error Rate

## 🔧 Advanced

### KenLM Language Model

```bash
# Install KenLM
pip install https://github.com/kpu/kenlm/archive/master.zip

# Train LM
lmplz -o 4 < corpus.txt > lm.arpa
build_binary lm.arpa lm.bin

# Use in inference
python inference.py --audio audio.wav --model model.pt --lm lm.bin
```

### Docker Deployment

```bash
# Build
docker build -t turkish-asr .

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -v ./runs:/app/models \
  -e ASR_MODEL_PATH=/app/models/best_model.pt \
  turkish-asr
```

## 📄 License

MIT License - See LICENSE file

---
*Developed by Muhammed Emin Korkut - Deep Zeka A.Ş*
