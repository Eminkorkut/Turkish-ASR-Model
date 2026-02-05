"""
Turkish ASR Inference Script

Supports:
- Single audio file transcription
- Batch processing
- Greedy and Beam Search decoding
"""

import torch
import argparse
import os
from pathlib import Path
from typing import Optional, List

from data.preprocessing import AudioPreprocessor
from data.tokenizer import TurkishTokenizer
from model.conformer import TurkishASRModel
from utils.decoding import CTCDecoder, NGramLanguageModel


class ASRInference:
    """
    ASR Inference Pipeline.
    
    Usage:
        asr = ASRInference("path/to/model.pt")
        text = asr.transcribe("audio.wav")
    """
    
    def __init__(
        self,
        model_path: str,
        n_mel_channels: int = 80,
        d_model: int = 256,
        n_heads: int = 4,
        n_blocks: int = 8,
        device: Optional[str] = None,
        use_beam_search: bool = False,
        beam_width: int = 10
    ):
        """
        Initialize ASR inference.
        
        Args:
            model_path: Path to trained model checkpoint
            n_mel_channels: Number of mel filterbanks
            d_model: Model dimension
            n_heads: Number of attention heads
            n_blocks: Number of Conformer blocks
            device: Device to use (auto-detect if None)
            use_beam_search: Use beam search decoding
            beam_width: Beam width for beam search
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize tokenizer
        self.tokenizer = TurkishTokenizer()
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            n_mels=n_mel_channels,
            normalize=True,
            device="cpu"  # Preprocessing on CPU
        )
        
        # Load model
        self.model = TurkishASRModel(
            n_mel_channels=n_mel_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            n_classes=self.tokenizer.vocab_size
        ).to(self.device)
        
        self._load_checkpoint(model_path)
        self.model.eval()
        
        # Decoder
        self.use_beam_search = use_beam_search
        if use_beam_search:
            lm = NGramLanguageModel()
            self.decoder = CTCDecoder(self.tokenizer, beam_width=beam_width, lm=lm)
        else:
            self.decoder = None
            
        print(f"ASR ready on {self.device}")
        
    def _load_checkpoint(self, path: str) -> None:
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded from: {path}")
        
    @torch.no_grad()
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe single audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            
        Returns:
            Transcribed text
        """
        # Extract features
        features = self.preprocessor(audio_path)
        features = features.unsqueeze(0).to(self.device)  # Add batch dim
        
        # Forward pass
        output = self.model(features)  # (1, T, C)
        logits = output[0]  # (T, C)
        
        # Decode
        if self.use_beam_search and self.decoder:
            text = self.decoder.decode(logits)
        else:
            # Greedy decoding
            pred_ids = torch.argmax(logits, dim=-1).tolist()
            text = self.tokenizer.ctc_decode(pred_ids)
            
        return text
    
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of transcribed texts
        """
        results = []
        for path in audio_paths:
            try:
                text = self.transcribe(path)
                results.append(text)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append("")
        return results


def main():
    parser = argparse.ArgumentParser(description="Turkish ASR Inference")
    parser.add_argument("--audio", type=str, required=True, help="Audio file or directory")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--beam_search", action="store_true", help="Use beam search decoding")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width")
    parser.add_argument("--n_mel_channels", type=int, default=80, help="Mel channels")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--n_blocks", type=int, default=8, help="Conformer blocks")
    
    args = parser.parse_args()
    
    # Initialize ASR
    asr = ASRInference(
        model_path=args.model,
        n_mel_channels=args.n_mel_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        use_beam_search=args.beam_search,
        beam_width=args.beam_width
    )
    
    # Process
    audio_path = Path(args.audio)
    
    if audio_path.is_dir():
        # Batch process directory
        audio_files = list(audio_path.glob("*.wav"))
        print(f"Found {len(audio_files)} audio files")
        
        for f in audio_files:
            text = asr.transcribe(str(f))
            print(f"{f.name}: {text}")
    else:
        # Single file
        text = asr.transcribe(str(audio_path))
        print(f"\nTranscription:\n{text}\n")


if __name__ == "__main__":
    main()
