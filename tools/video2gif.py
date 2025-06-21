import argparse
import cv2
from PIL import Image
from tqdm import tqdm
import os

def mp4_to_gif(input_path, output_path, start_time=None, end_time=None, resize=None, fps=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * video_fps) if start_time else 0
    end_frame = int(end_time * video_fps) if end_time else total_frames

    if fps is None:
        fps = video_fps

    frame_interval = int(video_fps // fps) if fps < video_fps else 1

    images = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    num_frames_to_process = (end_frame - start_frame) // frame_interval

    with tqdm(total=num_frames_to_process, desc="Processing frames") as pbar:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame - start_frame) % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                if resize:
                    img = img.resize(resize, Image.ANTIALIAS)
                images.append(img)
                pbar.update(1)

            current_frame += 1

    cap.release()

    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"GIF saved to {output_path}")
    else:
        print("No frames captured for GIF.")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert MP4 to GIF using OpenCV and PIL")
    parser.add_argument("input", help="Path to input MP4 video")
    parser.add_argument("output", help="Path to output GIF")
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("width", "height"), help="Resize to width height")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mp4_to_gif(
        input_path=args.input,
        output_path=args.output,
        start_time=args.start,
        end_time=args.end,
        resize=tuple(args.resize) if args.resize else None,
        fps=args.fps
    )