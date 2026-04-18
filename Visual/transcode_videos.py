#!/usr/bin/env python3
import json
import shutil
import subprocess
from pathlib import Path


def transcode_one(input_path: Path) -> Path | None:
    webm_out = input_path.with_suffix(".webm")
    if webm_out.exists():
        return webm_out
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    vf_candidates = ["scale='min(1280,iw)':-2", "scale=1280:-2", None]
    for vf in vf_candidates:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "0",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-an",
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd += [str(webm_out)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if webm_out.exists():
                return webm_out
        except Exception:
            continue
    return None

from tqdm import tqdm
def main():
    infer_path = Path("~/autodl-fs/output/libero/SFT_eval/val_infer.jsonl").expanduser()
    if not infer_path.exists():
        print(f"not found: {infer_path}")
        return
    vids = set()
    for line in infer_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        for v in obj.get("videos", []):
            vids.add(Path(v).expanduser())
    print(f"found {len(vids)} videos")
    ok = 0
    fail = 0
    for v in tqdm(sorted(vids)): 
        out = transcode_one(v)
        if out:
            ok += 1
        else:
            fail += 1
    print(f"done: ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
