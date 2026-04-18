from __future__ import annotations

import os
import sys
import argparse
import subprocess
from pathlib import Path

from tb_logger import TBLogger

DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "/root/autodl-tmp/data")
DEFAULT_OUTPUT_DIR = str(Path.home() / "autodl-fs" / "output" / "libero/SFT")
DEFAULT_CACHE_ROOT = Path.home() / "autodl-fs" / "cache" / "libero_qwenvl"
DEFAULT_MODELSCOPE_CACHE = str(DEFAULT_CACHE_ROOT / "modelscope")

def parse_args():
    p = argparse.ArgumentParser(description="Train IVY-xDETECTOR on FF++ videos via ms-swift")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--train_jsonl", type=str, default=str(Path(DEFAULT_DATA_ROOT) / "libero" / "train.jsonl"))
    p.add_argument("--val_jsonl", type=str, default=str(Path(DEFAULT_DATA_ROOT) / "libero" / "val.jsonl"))
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--gpu_ids", type=str, default="0")
    p.add_argument("--eval_and_save_steps", type=int, default=200)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--truncation_strategy", type=str, default="left")

    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--quant_bits", type=int, default=4)
    p.add_argument("--freeze_vit", type=str, default="false")

    p.add_argument("--video_max_pixels", type=int, default=80200)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--fps_max_frames", type=int, default=8)

    p.add_argument("--swift_path", type=str, default=None)
    p.add_argument("--tb_logdir", type=str, default=None)
    return p.parse_args()

def find_swift_executable(swift_path: str | None = None, tb: TBLogger | None = None) -> str:
    import shutil

    exe = shutil.which("swift")
    if exe:
        return exe
    repo_root = Path(__file__).parent.parent
    local_swift = repo_root / "ms-swift"
    if swift_path:
        local_swift = Path(swift_path)
    if local_swift.exists():
        if tb:
            tb.print(f"[INFO] Installing ms-swift from {local_swift} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(local_swift), "--quiet"])
        exe = shutil.which("swift")
        if exe:
            return exe
    raise RuntimeError("'swift' command not found. Please install ms-swift: pip install -e ../ms-swift")


def main():
    args = parse_args()
    tb = TBLogger(args.tb_logdir or (Path(args.output_dir) / "tb_train_wrapper"))

    for p in [args.train_jsonl, args.val_jsonl]:
        if not Path(p).exists():
            tb.print(f"[ERROR] Data file not found: {p}")
            tb.print("        Please run prepare_ffpp_video_dataset.py first.")
            tb.close()
            sys.exit(1)

    swift_exe = find_swift_executable(args.swift_path, tb=tb)
    tb.print(f"[INFO] Using swift: {swift_exe}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://hf-mirror.com")
    env["MODELSCOPE_CACHE"] = env.get("MODELSCOPE_CACHE", DEFAULT_MODELSCOPE_CACHE)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    env["VIDEO_MAX_PIXELS"] = str(args.video_max_pixels)
    env["FPS"] = str(args.fps)
    env["FPS_MAX_FRAMES"] = str(args.fps_max_frames)

    cmd = [
        swift_exe,
        "sft",
        "--model",
        args.model,
        "--dataset",
        args.train_jsonl,
        "--val_dataset",
        args.val_jsonl,
        "--tuner_type",
        "lora",
        "--torch_dtype",
        "bfloat16",
        "--num_train_epochs",
        str(args.epochs),
        "--per_device_train_batch_size",
        str(args.batch_size),
        "--per_device_eval_batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_rank * 2),
        "--lora_dropout",
        str(args.lora_dropout),
        "--target_modules",
        "all-linear",
        "--freeze_vit",
        str(args.freeze_vit).lower(),
        "--gradient_accumulation_steps",
        str(args.grad_accum),
        "--eval_steps",
        str(args.eval_and_save_steps),
        "--save_steps",
        str(args.eval_and_save_steps),
        "--save_total_limit",
        "3",
        "--logging_steps",
        "10",
        "--max_length",
        str(args.max_length),
        "--truncation_strategy",
        str(args.truncation_strategy),
        "--output_dir",
        args.output_dir,
        "--warmup_ratio",
        "0.1",
        "--dataloader_num_workers",
        "0",
        "--dataset_num_proc",
        "1",
        "--load_from_cache_file",
        "false",
        "--save_only_model",
        "false",
    ]

    if args.quant_bits and args.quant_bits > 0:
        cmd += ["--quant_bits", str(args.quant_bits), "--quant_method", "bnb"]
        tb.print(f"[INFO] QLoRA enabled: {args.quant_bits}-bit quantization")

    tb.print("[INFO] Running command:")
    tb.print("  " + " \\\n  ".join(cmd))
    tb.print()

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        tb.print(f"[ERROR] Training failed with return code {result.returncode}")
        tb.close()
        sys.exit(result.returncode)
    tb.print(f"[DONE] Training complete. Checkpoints: {args.output_dir}")
    tb.close()


if __name__ == "__main__":
    main()
