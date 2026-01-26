import torch
import torch.nn.functional as F
import argparse
import scipy.io.wavfile as wav
import numpy as np

# Proje modülleri
from utils.config import get_config
from data.tokenizer import TurkishTokenizer
from data.preprocessing import pre_emphasis, framing, windowing, power_spectrum, get_filter_banks, normalize_features
from model.conformer import TurkishASRModel
from utils.decoding import CTCDecoder, NGramLanguageModel

def preprocess_audio(wav_path, n_mel_channels=40):
    sample_rate, signal = wav.read(wav_path)
    signal = signal.astype(float)
    signal = pre_emphasis(signal)
    frames = framing(signal, sample_rate)
    frames = windowing(frames)
    pow_spec = power_spectrum(frames)
    features = get_filter_banks(pow_spec, sample_rate, nfilt=n_mel_channels)
    features = normalize_features(features)
    return torch.FloatTensor(features).unsqueeze(0) # (1, Time, Dim)

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tokenizer Yükle
    tokenizer = TurkishTokenizer(model_path="tokenizer_bpe.model") # Varsayılan model
    print(f"Tokenizer yüklendi. Vocab: {tokenizer.vocab_size}")
    
    # 2. Model Yükle
    model = TurkishASRModel(
        n_mel_channels=40, # Varsayılan
        d_model=256,
        n_heads=4,
        n_blocks=8, # Config ile aynı olmalı
        n_classes=tokenizer.vocab_size
    ).to(device)
    
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        # Checkpoint bazen tam state_dict bazen model objesi olabilir
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
        model.load_state_dict(state_dict, strict=False)
        print("Model ağırlıkları yüklendi.")
    
    model.eval()
    
    # 3. Ses İşle
    features = preprocess_audio(args.wav_path).to(device)
    
    # 4. Tahmin
    with torch.no_grad():
        output = model(features) # (1, T/4, Vocab)
        probs = output[0] # (T, V)
    
    # 5. Greedy Decode
    pred_ids = torch.argmax(probs, dim=-1).tolist()
    greedy_text = tokenizer.ctc_decode(pred_ids)
    print(f"\nGreedy Tahmin: {greedy_text}")
    
    # 6. Beam Search Decode
    # Basit bir LM simülasyonu (Opsiyonel)
    lm = NGramLanguageModel()
    # lm.train(["örnek cümleler"]) # Eğitilirse etkisi olur
    
    decoder = CTCDecoder(tokenizer, beam_width=args.beam_width, lm=lm)
    beam_text = decoder.decode(probs)
    print(f"Beam Tahmin:   {beam_text}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", type=str, required=True, help="Test edilecek ses dosyası")
    parser.add_argument("--model_path", type=str, required=True, help="Eğitilmiş model yolu (.pt)")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam search genişliği")
    args = parser.parse_args()
    
    predict(args)
