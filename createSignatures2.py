import cv2
import numpy as np

def downsample_frame(frame, scale_percent):
    """Reduce the size of the frame to the specified percentage."""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def calculate_histogram(frame):
    """Calculate a normalized color histogram for the given frame."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def calculate_motion(previous_frame, current_frame):
    """Calculate the magnitude of motion between two frames using optical flow."""
    flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.sum(magnitude)

def calculate_optical_flows(frames):
    # Initialize list to hold motion magnitudes
    motion_magnitudes = []
    prev_frame_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        next_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitude = np.mean(magnitude)
        motion_magnitudes.append(motion_magnitude)
        prev_frame_gray = next_frame_gray

    return motion_magnitudes


def detect_shot_boundaries(video_path, color_threshold, motion_threshold, downsample_percent):
    """Detect shot boundaries in a video and return the start and end frames of each shot, 
    as well as the average color histogram for each shot."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't read video file.")
        return [], []

    # Initialize variables for the previous frame
    previous_frame = downsample_frame(frame, downsample_percent)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_hist = calculate_histogram(previous_frame)
    shot_boundaries = []
    histograms = []
    shot_frames = []
    current_shot_histograms = []  # To store histograms of the current shot
    motion_scores = []
    current_shot_start = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Append the last shot when the video ends
            if current_shot_histograms:
                # Calculate the average histogram of the current shot
                avg_hist = np.mean(current_shot_histograms, axis=0)
                histograms.append(avg_hist)
            shot_boundaries.append((current_shot_start, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1))

            if shot_frames:
                motion_score = calculate_optical_flows(shot_frames)
                motion_scores.append(motion_score)
                
            break
        
        # Downsample and process the current frame
        frame_downsampled = downsample_frame(frame, downsample_percent)
        shot_frames.append(frame_downsampled)
        frame_gray = cv2.cvtColor(frame_downsampled, cv2.COLOR_BGR2GRAY)
        current_hist = calculate_histogram(frame_downsampled)
        current_shot_histograms.append(current_hist)  # Add histogram to current shot

        # Calculate histogram difference and motion magnitude
        hist_diff = cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA)
        motion_value = calculate_motion(previous_frame_gray, frame_gray)

        # Check for shot boundary
        if hist_diff > color_threshold or motion_value > motion_threshold:
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Append the previous shot (start frame, end frame) and its average histogram
            if current_shot_histograms:
                avg_hist = np.mean(current_shot_histograms, axis=0)
                histograms.append(avg_hist)
            shot_boundaries.append((current_shot_start, frame_number - 1))
            motion_score = calculate_optical_flows(shot_frames)
            motion_scores.append(motion_score)
            shot_frames = []
            # Start a new shot
            current_shot_start = frame_number
            current_shot_histograms = []  # Reset for the new shot

        # Update the previous frame information
        previous_frame_gray = frame_gray
        previous_hist = current_hist

    cap.release()

    return shot_boundaries, histograms, motion_scores

# Parameters
video_path = 'Data/Videos/video2.mp4'
color_threshold = 0.5  # Experimentally determined threshold for color change
motion_threshold = 2.5e5  # Experimentally determined threshold for motion change
downsample_percent = 60  # Downsample frames by 60% to reduce computation

# Detect shot boundaries and print them
boundaries, shot_histograms, motion_scores = detect_shot_boundaries(video_path, color_threshold, motion_threshold, downsample_percent)
print(len(boundaries))
print(len(shot_histograms))
print(len(motion_scores))
