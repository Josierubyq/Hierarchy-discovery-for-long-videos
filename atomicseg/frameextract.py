#!/usr/bin/env python3
# frameextract_ffmpeg.py
"""
基于 ffmpeg + 多进程并行的断点续跑抽帧脚本。
支持：
  - 跳过已提取完的视频目录
  - ffmpeg 子进程放在独立进程组，响应 Ctrl+C
  - 容错 I/O 错误、同名文件冲突
Usage:
  python3 frameextract_ffmpeg.py \
    --video_root "/media/wang/.../kk" \
    --frame_root "/media/wang/.../kkframe" \
    --workers 8
"""
import os
import signal
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing


def extract_with_ffmpeg(video_path: Path, frame_dir: Path):
    """
    用 ffmpeg 从视频中抽帧到指定目录，支持续跑与跳过已完成。
    """
    # 冲突检查：如果目标存在同名普通文件则跳过
    try:
        if frame_dir.exists() and not frame_dir.is_dir():
            print(f"[WARN] 目标存在同名文件，跳过视频: {video_path.name}")
            return
    except OSError as e:
        print(f"[WARN] 检查 {frame_dir} 时出错，跳过: {e}")
        return

    # 如果已提取帧数 >= 总帧数，则跳过
    try:
        # 统计已有帧文件数量
        existing = list(frame_dir.glob("*.jpg"))
        start_idx = len(existing)
    except Exception:
        start_idx = 0

    # 获取总帧数 via ffprobe
    cmd_probe = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=nb_frames', '-of', 'default=nokey=1:noprint_wrappers=1', str(video_path)
    ]
    try:
        total = subprocess.check_output(cmd_probe, stderr=subprocess.DEVNULL)
        total_frames = int(total.strip())
    except Exception:
        total_frames = None

    if total_frames is not None and start_idx >= total_frames:
        print(f"[SKIP] 已完全抽取过: {video_path.name} ({start_idx}/{total_frames})")
        return

    # 创建输出目录
    try:
        frame_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[ERROR] 无法创建目录 {frame_dir}, 跳过: {e}")
        return

    # 构建 ffmpeg 命令
    # -start_number 用于续跑
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-start_number", str(start_idx),
        "-q:v", "2",
        str(frame_dir / "%06d.jpg")
    ]

    # 启动子进程于新进程组
    proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
    try:
        proc.wait()
    except KeyboardInterrupt:
        # Ctrl+C 时杀掉整个子进程组
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        raise


def main(video_root: Path, frame_root: Path, workers: int):
    if not video_root.is_dir():
        raise ValueError(f"视频根目录不存在: {video_root}")

    videos = [p for p in video_root.rglob("*.mp4")]
    print(f"共发现 {len(videos)} 个视频，启动 {workers} 个进程并行抽帧…")

    tasks = []
    for vid in videos:
        rel = vid.relative_to(video_root).with_suffix("")
        out_dir = frame_root / rel
        tasks.append((vid, out_dir))

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(extract_with_ffmpeg, vid, out): vid for vid, out in tasks}
        try:
            for _ in tqdm(as_completed(futures), total=len(futures), unit="video"):
                pass
        except KeyboardInterrupt:
            print("\n检测到中断，取消剩余任务…")
            exe.shutdown(wait=False, cancel_futures=True)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并行断点续跑视频抽帧脚本")
    parser.add_argument("--video_root", type=Path, required=True,
                        help="源视频根目录，例如 /media/.../kk")
    parser.add_argument("--frame_root", type=Path, required=True,
                        help="保存帧的根目录，例如 /media/.../kkframe")
    parser.add_argument("--workers", type=int,
                        default=max(1, multiprocessing.cpu_count()-1),
                        help="并行进程数，默认 CPU 核数-1")
    args = parser.parse_args()
    main(args.video_root, args.frame_root, args.workers)
