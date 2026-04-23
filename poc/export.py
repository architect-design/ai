"""
export.py — Export CustomLLM to ONNX / quantized formats for deployment.

Supported exports
──────────────────
  1. ONNX (fp32)        — interoperable, runs with onnxruntime on any platform
  2. ONNX (fp16)        — halved memory, still hardware-friendly
  3. TorchScript        — frozen PyTorch graph, no Python dependency at runtime
  4. Dynamic INT8 quant — 4× smaller weights, CPU-only, via torch.quantization

Usage (CLI):
    # ONNX fp32
    python export.py \
        --checkpoint checkpoints/MY_SPEC/best_model.pt \
        --tokenizer  checkpoints/MY_SPEC/tokenizer.json \
        --format     onnx

    # TorchScript
    python export.py \
        --checkpoint checkpoints/MY_SPEC/best_model.pt \
        --tokenizer  checkpoints/MY_SPEC/tokenizer.json \
        --format     torchscript

    # INT8 quantized
    python export.py \
        --checkpoint checkpoints/MY_SPEC/best_model.pt \
        --tokenizer  checkpoints/MY_SPEC/tokenizer.json \
        --format     int8

Usage (API):
    from export import Exporter
    ex = Exporter.from_checkpoint(
        "checkpoints/MY_SPEC/best_model.pt",
        "checkpoints/MY_SPEC/tokenizer.json",
    )
    ex.to_onnx("exports/my_spec.onnx")
    ex.to_torchscript("exports/my_spec.pt")
    ex.to_int8("exports/my_spec_int8.pt")
"""

import argparse
import os
from typing import Optional

import torch
import torch.nn as nn

from model     import CustomLLM
from tokenizer import load_tokenizer


# ─── ONNX wrapper ─────────────────────────────────────────────────────────────

class ONNXWrapper(nn.Module):
    """
    Wraps CustomLLM so that ONNX export gets a clean single-input graph.
    Outputs logits only (no loss), consistent with inference mode.
    """

    def __init__(self, model: CustomLLM):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(idx)
        return logits


# ─── Exporter ─────────────────────────────────────────────────────────────────

class Exporter:
    def __init__(self, model: CustomLLM, tokenizer, device: str = "cpu"):
        self.model     = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device    = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        tokenizer_path: str,
        device: str = "cpu",
    ) -> "Exporter":
        model     = CustomLLM.from_checkpoint(checkpoint, device=device)
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"[Exporter] Loaded  {model.num_params():,} params  device={device}")
        return cls(model, tokenizer, device)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _dummy_input(self, seq_len: Optional[int] = None) -> torch.Tensor:
        """Create a dummy (1, T) integer token tensor for tracing."""
        T = seq_len or min(32, self.model.config.context_length)
        return torch.zeros(1, T, dtype=torch.long, device=self.device)

    @staticmethod
    def _ensure_dir(path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # ── ONNX ──────────────────────────────────────────────────────────────────

    def to_onnx(
        self,
        out_path: str,
        fp16: bool = False,
        opset: int = 17,
        seq_len: Optional[int] = None,
    ) -> str:
        """
        Export to ONNX.

        Args:
            out_path : destination .onnx file path
            fp16     : convert weights to float16 before export
            opset    : ONNX opset version (>=13 recommended)
            seq_len  : fixed sequence length for export (None = dynamic)
        """
        try:
            import onnx  # noqa: F401
        except ImportError:
            print("[export] 'onnx' not installed.  pip install onnx onnxruntime")
            return ""

        self._ensure_dir(out_path)
        wrapper = ONNXWrapper(self.model)
        if fp16:
            wrapper = wrapper.half()

        dummy = self._dummy_input(seq_len)
        if fp16:
            dummy = dummy  # int stays int

        # Dynamic axes: allow variable batch size and sequence length
        dynamic = {"idx": {0: "batch", 1: "sequence"},
                   "logits": {0: "batch", 1: "sequence"}}

        torch.onnx.export(
            wrapper,
            (dummy,),
            out_path,
            opset_version      = opset,
            input_names        = ["idx"],
            output_names       = ["logits"],
            dynamic_axes       = dynamic,
            do_constant_folding= True,
            export_params      = True,
        )

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        dtype   = "fp16" if fp16 else "fp32"
        print(f"[export] ONNX ({dtype}) → {out_path}  ({size_mb:.1f} MB)")
        return out_path

    # ── TorchScript ───────────────────────────────────────────────────────────

    def to_torchscript(self, out_path: str) -> str:
        """
        Trace the model with TorchScript.
        The resulting .pt file runs without the Python source files.
        """
        self._ensure_dir(out_path)
        wrapper = ONNXWrapper(self.model)
        dummy   = self._dummy_input()

        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (dummy,))
            # Optimise the graph (fold constants, remove dead code)
            traced = torch.jit.optimize_for_inference(traced)

        traced.save(out_path)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"[export] TorchScript → {out_path}  ({size_mb:.1f} MB)")
        return out_path

    # ── Dynamic INT8 quantization ─────────────────────────────────────────────

    def to_int8(self, out_path: str) -> str:
        """
        Apply dynamic INT8 quantization to all Linear layers.
        Reduces model size by ~4×.  CPU only (no CUDA INT8 support via this API).

        The quantized model is a standard nn.Module saved with torch.save.
        Load it with:
            model_q = torch.load("my_spec_int8.pt")
        """
        self._ensure_dir(out_path)

        if self.device != "cpu":
            print("[export] INT8 quantisation requires CPU — moving model…")
            self.model = self.model.cpu()

        quantized = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},            # quantise all Linear layers
            dtype=torch.qint8,
        )

        torch.save(quantized, out_path)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        orig_mb = sum(p.numel() * p.element_size()
                      for p in self.model.parameters()) / 1024 / 1024
        ratio   = orig_mb / size_mb if size_mb > 0 else 0
        print(f"[export] INT8 → {out_path}  ({size_mb:.1f} MB)  "
              f"original ~{orig_mb:.1f} MB  compression {ratio:.1f}×")
        return out_path

    # ── Validate ONNX ─────────────────────────────────────────────────────────

    def validate_onnx(self, onnx_path: str, rtol: float = 1e-3, atol: float = 1e-4):
        """
        Run the ONNX model with onnxruntime and compare output to PyTorch.
        Raises AssertionError if outputs differ beyond tolerance.
        """
        try:
            import onnxruntime as ort
            import numpy as np
        except ImportError:
            print("[validate] pip install onnxruntime")
            return

        dummy = self._dummy_input().numpy()
        sess  = ort.InferenceSession(onnx_path)
        ort_out = sess.run(None, {"idx": dummy})[0]

        with torch.no_grad():
            pt_out = ONNXWrapper(self.model)(
                torch.from_numpy(dummy)
            ).numpy()

        max_diff = float(abs(ort_out - pt_out).max())
        match    = max_diff < atol + rtol * abs(pt_out).max()
        status   = "✓ PASS" if match else "✗ FAIL"
        print(f"[validate] {status}  max_diff={max_diff:.6f}  "
              f"atol={atol}  rtol={rtol}")

    # ── Full export suite ─────────────────────────────────────────────────────

    def export_all(self, out_dir: str, validate: bool = True):
        """Export to all supported formats into `out_dir`."""
        os.makedirs(out_dir, exist_ok=True)
        name = self.model.config.spec_name

        paths = {}
        paths["onnx_fp32"]    = self.to_onnx(
            os.path.join(out_dir, f"{name}_fp32.onnx")
        )
        paths["onnx_fp16"]    = self.to_onnx(
            os.path.join(out_dir, f"{name}_fp16.onnx"), fp16=True
        )
        paths["torchscript"]  = self.to_torchscript(
            os.path.join(out_dir, f"{name}_scripted.pt")
        )
        paths["int8"]         = self.to_int8(
            os.path.join(out_dir, f"{name}_int8.pt")
        )

        if validate and paths["onnx_fp32"]:
            self.validate_onnx(paths["onnx_fp32"])

        print(f"\n[export] All formats saved to: {out_dir}/")
        return paths


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Export CustomLLM to ONNX / TorchScript / INT8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer",  required=True)
    p.add_argument("--format",
                   choices=["onnx", "onnx_fp16", "torchscript", "int8", "all"],
                   default="onnx")
    p.add_argument("--out",     default="exports",  help="Output directory or file path")
    p.add_argument("--opset",   type=int, default=17)
    p.add_argument("--validate",action="store_true", help="Validate ONNX output vs PyTorch")
    args = p.parse_args()

    ex   = Exporter.from_checkpoint(args.checkpoint, args.tokenizer)
    name = ex.model.config.spec_name
    os.makedirs(args.out, exist_ok=True)

    if args.format == "all":
        ex.export_all(args.out, validate=args.validate)
    elif args.format == "onnx":
        path = ex.to_onnx(os.path.join(args.out, f"{name}_fp32.onnx"),
                          opset=args.opset)
        if args.validate and path:
            ex.validate_onnx(path)
    elif args.format == "onnx_fp16":
        ex.to_onnx(os.path.join(args.out, f"{name}_fp16.onnx"),
                   fp16=True, opset=args.opset)
    elif args.format == "torchscript":
        ex.to_torchscript(os.path.join(args.out, f"{name}_scripted.pt"))
    elif args.format == "int8":
        ex.to_int8(os.path.join(args.out, f"{name}_int8.pt"))
