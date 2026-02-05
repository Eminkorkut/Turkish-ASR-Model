"""
ONNX Export Script for Turkish ASR Model.

Exports the trained model to ONNX format for deployment.
"""

import torch
import argparse
import os
from pathlib import Path

from model.conformer import TurkishASRModel
from data.tokenizer import TurkishTokenizer


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    n_mel_channels: int = 80,
    d_model: int = 256,
    n_heads: int = 4,
    n_blocks: int = 8,
    opset_version: int = 14,
    dynamic_axes: bool = True
) -> None:
    """
    Export trained ASR model to ONNX format.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        output_path: Output ONNX file path
        n_mel_channels: Model mel channels
        d_model: Model dimension
        n_heads: Attention heads
        n_blocks: Conformer blocks
        opset_version: ONNX opset version
        dynamic_axes: Use dynamic axes for batch and sequence length
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load tokenizer to get vocab size
    tokenizer = TurkishTokenizer()
    n_classes = tokenizer.vocab_size
    
    # Create model
    model = TurkishASRModel(
        n_mel_channels=n_mel_channels,
        d_model=d_model,
        n_heads=n_heads,
        n_blocks=n_blocks,
        n_classes=n_classes
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    batch_size = 1
    seq_len = 100
    dummy_input = torch.randn(batch_size, seq_len, n_mel_channels)
    dummy_lengths = torch.tensor([seq_len])
    
    # Configure dynamic axes
    if dynamic_axes:
        axes = {
            "input_features": {0: "batch_size", 1: "sequence_length"},
            "output_logits": {0: "batch_size", 1: "output_length"}
        }
    else:
        axes = None
    
    # Export
    print(f"Exporting to ONNX (opset {opset_version})...")
    
    # Wrapper to handle input_lengths as optional for ONNX
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # ONNX export without masking for simplicity
            return self.model(x, input_lengths=None)
    
    wrapped_model = ONNXWrapper(model)
    
    torch.onnx.export(
        wrapped_model,
        (dummy_input,),
        output_path,
        input_names=["input_features"],
        output_names=["output_logits"],
        dynamic_axes=axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: PASSED")
    except ImportError:
        print("Install 'onnx' package to verify the exported model.")
    except Exception as e:
        print(f"ONNX verification failed: {e}")
        
    # Print model info
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export ASR model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--n_mel_channels", type=int, default=80, help="Mel channels")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--n_blocks", type=int, default=8, help="Conformer blocks")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    
    args = parser.parse_args()
    
    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        n_mel_channels=args.n_mel_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        opset_version=args.opset
    )


if __name__ == "__main__":
    main()
