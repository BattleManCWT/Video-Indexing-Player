import numpy as np
from skimage.metrics import structural_similarity as ssim

def read_frame(file, frame_num, width, height):
    frame_size = width * height * 3  # 3 bytes per pixel for RGB
    with open(file, 'rb') as f:
        f.seek(frame_num * frame_size)
        frame = np.fromfile(f, dtype=np.uint8, count=frame_size)
        frame = frame.reshape((height, width, 3))
    return frame

def compare_frames(frame_a, frame_b):
    # Convert frames to grayscale as SSIM is typically computed on grayscale images
    frame_a_gray = np.dot(frame_a[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    frame_b_gray = np.dot(frame_b[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    return ssim(frame_a_gray, frame_b_gray, data_range=frame_a_gray.max() - frame_a_gray.min())

def locate_exact_frame(query_rgb, video_rgb, approximate_location, search_range = 10):
    # Read the first frame of video A
    query_first_frame = read_frame(query_rgb, 0, width, height)
    frame_num = approximate_location - search_range
    max_frame_num = approximate_location + search_range
    best_match = 0
    highest_ssim = 0

    try:
        while frame_num <= max_frame_num:
            # Read each frame of video B
            video_frame = read_frame(video_rgb, frame_num, width, height)
            similarity = compare_frames(query_first_frame, video_frame)

            if similarity > highest_ssim:
                highest_ssim = similarity
                best_match = frame_num

            frame_num += 1

    except FileNotFoundError:
        # No more frames to read
        pass

    return best_match



width, height = 352,288  # Replace with your video resolution
print(locate_exact_frame("Data/Videos/rgb/video9_1.rgb", "Data/Videos/rgb/video9.rgb", 5250, 5))




