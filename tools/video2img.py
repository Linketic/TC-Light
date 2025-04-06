# read video(.mp4, .gif, .avi, etc) and turn it to images with appointed sampling interval

import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm

def video2img(input_video, output_folder, sampling_interval):
    # create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # read video
    cap = cv2.VideoCapture(input_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {frame_count}, FPS: {fps}")

    video_name = os.path.basename(input_video).split('.')[0]

    for i in tqdm(range(0, frame_count), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read frame from video")
        # save frame as image
        if i % sampling_interval == 0:
            img_name = os.path.join(output_folder, f"{video_name}_{i:04d}.jpg")
            cv2.imwrite(img_name, frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_video", help="input video file")
    parser.add_argument("-o", "--output_folder", help="output folder to save images")
    parser.add_argument("-s", "--sampling_interval", type=float, default=1.0, help="sampling interval in seconds")
    args = parser.parse_args()

    video2img(args.input_video, args.output_folder, args.sampling_interval)