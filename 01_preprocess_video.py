import os, subprocess
from os import path
from pathlib import Path
import torch
import clip
from PIL import Image
from tqdm import tqdm
import argparse

print(f"✅ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
# Example python
# uv run preprocess_video.py --video_path NExTVideo --output_dir pipeline_output --preprocessed_dir preprocessed_frames --fps 10


parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, default="NExTVideo")
parser.add_argument("--output_dir", type=str, default="pipeline_output")
parser.add_argument("--preprocessed_dir", type=str, default="preprocessed_frames")
parser.add_argument("--fps", type=int, default=10)

args = parser.parse_args()

VIDEO_PATH = args.video_path
OUTPUT_DIR = args.output_dir
PREPROCESSED_DIR = args.preprocessed_dir
FPS = args.fps
CLEAN_FRAMES = False
REUSE_EXISTING_FRAMES = True

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_frames(video_path):
    video_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi', '.mov'))])
    # print(f"🎬 Found {len(video_files)} video(s) in '{video_path}': {video_files}")
    for video_file in tqdm(video_files, desc="Extracting frames"):
        vid_path = Path(path.join(video_path, video_file))
        video_name = vid_path.stem
        temp_path = video_path + f"_{FPS}FPS"
        output_dir = Path(OUTPUT_DIR) / temp_path / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if CLEAN_FRAMES:
            # print("Clear old frames...")
            for p in output_dir.glob('*.jpg'):
                p.unlink()

        frame_files = sorted(list(output_dir.glob('*.jpg')))
        if frame_files and REUSE_EXISTING_FRAMES:
            # print(f"✅ Reusing existing {len(frame_files)} frames")
            pass
        else:
            # print("Using FFmpeg to extract frames...")
            cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-i', str(vid_path),
                '-vf', f'fps={FPS}',
                '-q:v', '2',
                str(output_dir / '%06d.jpg')
            ]
            subprocess.run(cmd, check=True)
            frame_files = sorted(list(output_dir.glob('*.jpg')))

def get_filenames(dir_path):
    return [f for f in os.listdir(dir_path)]


if __name__ == "__main__":
    extract_frames(VIDEO_PATH)
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    frame_filenames = sorted(get_filenames(path.join(OUTPUT_DIR, VIDEO_PATH + f"_{FPS}FPS")))
    
    for f in tqdm(frame_filenames, desc="Processing videos"):
        images = []
        frame_files = get_filenames(path.join(OUTPUT_DIR, VIDEO_PATH + f'_{FPS}FPS', f))
        for ff in tqdm(frame_files, desc=f"Processing frames for {f}", leave=False):
            try:
                with Image.open(path.join(OUTPUT_DIR, VIDEO_PATH + f'_{FPS}FPS', f, ff)) as img:
                    images.append(preprocess(img.convert("RGB")).to(device))
            except Exception as e:
                print(f"⚠️  Skip corrupted frame: {ff}")
                continue
        
        # make images from list to tensor
        images = torch.stack(images)
        output_dir = Path(PREPROCESSED_DIR) / (VIDEO_PATH + f"_{FPS}FPS")
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(images, path.join(output_dir, f"{f}.pt"))