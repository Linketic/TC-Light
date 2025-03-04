# convert a series of images to a video, with a given frame rate
# Usage: python img2video.py -i <input_folder> -o <output_video> -f <frame_rate>
# Example: python img2video.py -i ./images -o ./video.mp4 -f 30

import cv2
import os
import argparse
import imageio

def img2video(input_folder, output_video, frame_rate, max_length, save_images):
    # get the list of images
    images = [img for img in os.listdir(input_folder) if (img.endswith(".png") or img.endswith(".jpg"))]
    images.sort()
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape
    H = height if height < max_length else max_length
    W = width if width < max_length else max_length

    if save_images:
        if not os.path.exists(input_folder + "_cropped"):
            os.makedirs(input_folder + "_cropped")
        cropped_folder = input_folder + "_cropped"
    
    # create a video writer
    if output_video.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_video, fourcc, frame_rate, (W, H))

        # write images to video
        for image in images:
            img = cv2.imread(os.path.join(input_folder, image))
            if height > max_length:
                img = img[height//2 - max_length//2:height//2 + max_length//2, :, :]
            if width > max_length:
                img = img[:, width//2 - max_length//2:width//2 + max_length//2, :]
            video.write(img)

            if save_images:
                cv2.imwrite(os.path.join(cropped_folder, image), img)

        cv2.destroyAllWindows()
        video.release()

    elif output_video.endswith(".gif"):
        frames = []
        for image in images:
            img = cv2.imread(os.path.join(input_folder, image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if height > max_length:
                img = img[height//2 - max_length//2:height//2 + max_length//2, :, :]
            if width > max_length:
                img = img[:, width//2 - max_length//2:width//2 + max_length//2, :]
            frames.append(img)
        imageio.mimsave(output_video, frames, 'GIF', fps=frame_rate, loop=0)
    
    else:
        print("Unsupported file format. Please use .mp4 or .gif")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="input folder containing images")
    parser.add_argument("-o", "--output_video", help="output video file")
    parser.add_argument("-f", "--frame_rate", help="frame rate of the video")
    parser.add_argument("-m", "--max_length", help="crop video to this length", default=512)
    parser.add_argument("-s", "--save_images", help="save images to disk", action="store_true")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_video = args.output_video
    frame_rate = int(args.frame_rate)
    max_length = int(args.max_length)
    save_images = args.save_images

    img2video(input_folder, output_video, frame_rate, max_length, save_images)
    print("Video created successfully!")