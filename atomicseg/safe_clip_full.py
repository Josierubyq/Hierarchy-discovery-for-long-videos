import os, sys, gc, time
from PIL import Image
import torch, clip
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ========== 配置区 ==========
FRAME_DIR         = "/media/wang/New Volume/josie/GEBD-main/kinetics-dataset/kkframe"
OUTPUT_DIR        = "clip_embeddings_by_video"  # 输出目录
ORIG_FPS          = 30
TARGET_FPS        = 10
BATCH_SIZE        = 8         # 测试验证过的安全大小
NUM_WORKERS       = 1         # 测试验证过的安全设置

# 测试模式
TEST_MODE         = True      # 先设为True，测试成功后改为False
MAX_TEST_VIDEOS   = 1000      # 处理前1000个视频
# ============================

class VideoFrameDataset(Dataset):
    def __init__(self, frame_paths, preprocess):
        self.frame_paths = frame_paths
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.frame_paths[idx]).convert("RGB")
            return self.preprocess(img), self.frame_paths[idx]
        except Exception as e:
            print(f"⚠️ 图片读取失败: {self.frame_paths[idx]}")
            # 返回黑色图片
            black_img = Image.new('RGB', (224, 224), (0, 0, 0))
            return self.preprocess(black_img), self.frame_paths[idx]

def get_video_directories():
    """获取所有视频目录"""
    video_dirs = []
    
    if not os.path.exists(FRAME_DIR):
        print(f"❌ 目录不存在: {FRAME_DIR}")
        return []
    
    # 扫描所有子目录（每个子目录代表一个视频）
    for item in os.listdir(FRAME_DIR):
        video_path = os.path.join(FRAME_DIR, item)
        if os.path.isdir(video_path):
            # 检查目录中是否有图片文件
            has_images = False
            for file in os.listdir(video_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    has_images = True
                    break
            
            if has_images:
                video_dirs.append(video_path)
    
    video_dirs.sort()
    print(f"📁 找到 {len(video_dirs)} 个视频目录")
    
    # 测试模式只处理前几个
    if TEST_MODE:
        video_dirs = video_dirs[:MAX_TEST_VIDEOS]
        print(f"🧪 测试模式：只处理前 {len(video_dirs)} 个视频")
    
    return video_dirs

def get_frames_from_video_dir(video_dir):
    """从视频目录获取帧文件列表"""
    frame_paths = []
    
    for file in os.listdir(video_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            frame_paths.append(os.path.join(video_dir, file))
    
    frame_paths.sort()
    
    # 按FPS采样
    step = max(1, ORIG_FPS // TARGET_FPS)
    sampled_frames = frame_paths[::step]
    
    return sampled_frames

def process_single_video(video_dir, model, preprocess, device):
    """处理单个视频的所有帧"""
    video_name = os.path.basename(video_dir)
    
    # 获取该视频的所有帧
    frame_paths = get_frames_from_video_dir(video_dir)
    
    if not frame_paths:
        return None
    
    # 创建数据集和加载器
    dataset = VideoFrameDataset(frame_paths, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # 提取特征
    all_embeddings = []
    frame_names = []
    
    with torch.no_grad():
        # 使用position=1让单视频进度条显示在总进度条下方
        pbar = tqdm(dataloader, desc=f"  🎯 {video_name[:25]}", leave=False, position=1)
        
        for batch_idx, (batch_imgs, batch_paths) in enumerate(pbar):
            try:
                # GPU推理
                batch_imgs = batch_imgs.to(device, non_blocking=True)
                features = model.encode_image(batch_imgs)
                features = features / features.norm(dim=-1, keepdim=True)
                
                # 移回CPU
                features_cpu = features.cpu()
                all_embeddings.append(features_cpu)
                
                # 记录帧名称
                for path in batch_paths:
                    frame_name = os.path.basename(path)
                    frame_names.append(frame_name)
                
                # 清理GPU内存
                del batch_imgs, features
                torch.cuda.empty_cache()
                
                # 更新进度
                pbar.set_postfix({
                    'frames': len(frame_names)
                })
                
            except Exception as e:
                continue
        
        pbar.close()  # 关闭单视频进度条
    
    # 合并特征
    if all_embeddings:
        video_embeddings = torch.cat(all_embeddings, dim=0)
        
        # 保存结果
        output_data = {
            'embeddings': video_embeddings,
            'frame_names': frame_names,
            'video_name': video_name,
            'num_frames': len(frame_names)
        }
        
        return output_data
    else:
        return None

def main():
    print("🚀 开始按视频提取CLIP特征...")
    
    # 1. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 输出目录: {OUTPUT_DIR}")
    
    # 2. CUDA设置
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    print(f"🔧 GPU: {torch.cuda.get_device_name()}")
    
    # 3. 加载模型
    print("📦 加载CLIP模型...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # 4. 获取所有视频目录
    video_dirs = get_video_directories()
    if not video_dirs:
        print("❌ 没有找到视频目录")
        return
    
    # 5. 逐个处理视频
    successful_videos = 0
    failed_videos = 0
    
    # 添加总的进度条
    main_pbar = tqdm(video_dirs, desc="🎬 总进度", position=0)
    
    for i, video_dir in enumerate(main_pbar):
        video_name = os.path.basename(video_dir)
        
        # 更新总进度条描述
        main_pbar.set_description(f"🎬 总进度 [{i+1}/{len(video_dirs)}]")
        main_pbar.set_postfix({
            '成功': successful_videos,
            '失败': failed_videos,
            '当前': video_name[:30] + ('...' if len(video_name) > 30 else '')
        })
        
        try:
            # 检查是否已经处理过
            output_path = os.path.join(OUTPUT_DIR, f"{video_name}.pt")
            if os.path.exists(output_path):
                print(f"\n✅ 跳过已处理的视频: {video_name}")
                successful_videos += 1  # 已处理的也算成功
                continue
            
            # 处理视频（不显示单独的进度信息）
            result = process_single_video(video_dir, model, preprocess, device)
            
            if result:
                # 保存结果
                torch.save(result, output_path)
                successful_videos += 1
            else:
                failed_videos += 1
                
        except Exception as e:
            print(f"\n❌ 视频 {video_name} 处理失败: {e}")
            failed_videos += 1
            continue
    
    # 关闭总进度条
    main_pbar.close()
    
    # 6. 总结
    print(f"\n🎉 处理完成!")
    print(f"✅ 成功: {successful_videos} 个视频")
    print(f"❌ 失败: {failed_videos} 个视频")
    print(f"📁 结果保存在: {OUTPUT_DIR}/")
    
    # 7. 显示输出文件
    print("\n📋 生成的文件:")
    for file in sorted(os.listdir(OUTPUT_DIR)):
        if file.endswith('.pt'):
            file_path = os.path.join(OUTPUT_DIR, file)
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  {file} ({file_size:.1f}MB)")
    
    # 8. 清理
    torch.cuda.empty_cache()
    print("\n🧹 清理完成")

def load_video_features(video_name):
    """示例：如何加载单个视频的特征"""
    file_path = os.path.join(OUTPUT_DIR, f"{video_name}.pt")
    if os.path.exists(file_path):
        data = torch.load(file_path)
        print(f"视频: {data['video_name']}")
        print(f"帧数: {data['num_frames']}")
        print(f"特征维度: {data['embeddings'].shape}")
        print(f"帧名称示例: {data['frame_names'][:5]}")
        return data
    else:
        print(f"文件不存在: {file_path}")
        return None

if __name__ == "__main__":
    main()
    
    # 使用示例
    print("\n" + "="*50)
    print("💡 使用示例:")
    print("# 加载单个视频的特征")
    print("data = torch.load('clip_embeddings_by_video/视频名.pt')")
    print("embeddings = data['embeddings']  # 特征矩阵")
    print("frame_names = data['frame_names']  # 帧文件名列表")
    print("video_name = data['video_name']    # 视频名称")