import os
from os import path
from pathlib import Path
import torch
import clip
from tqdm import tqdm

print(f"✅ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    


VIDEO_PATH = "NExTVideo"
OUTPUT_DIR = "pipeline_output"
PREPROCESSED_DIR = "preprocessed_frames"
FEATURES_DIR = "features"
FPS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

    
def get_filenames(dir_path):
    return [f for f in os.listdir(dir_path)]


if __name__ == "__main__":
    torch_files = get_filenames(path.join(PREPROCESSED_DIR, VIDEO_PATH + f"_{FPS}FPS"))

    model, preprocess = clip.load("ViT-B/32", device=device)
    output_dir = Path(FEATURES_DIR) / (VIDEO_PATH + f"_{FPS}FPS")
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for f in tqdm(torch_files, desc="Encoding frames"):
            images = torch.load(path.join(PREPROCESSED_DIR, VIDEO_PATH + f"_{FPS}FPS", f))
            image_features = model.encode_image(images.to(device))
            torch.save(image_features.cpu(), output_dir / f)