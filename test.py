import torch
from tqdm import tqdm

print(f"✅ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # a = torch.randn(10000, 10000).to(device)
    # b = torch.randn(10000, 10000).to(device)
    for _ in tqdm(range(10000)):
        a = torch.randn(1000, 1000).to(device)
        b = torch.randn(1000, 1000).to(device)
        c = torch.matmul(a, b)  # This runs on GPU if device is 'cuda'