import cv2
import numpy as np
from pathlib import Path
import json

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

def process_shot(shot_frames, current_shot_histograms):
    """Process the accumulated frames of a shot to calculate the average histogram and motion score."""
    avg_hist = np.mean(np.array(current_shot_histograms), axis=0).tolist() if current_shot_histograms else [0] * (8 * 8 * 8)
    motion_scores = calculate_optical_flows(shot_frames)
    motion_score = np.mean(motion_scores).tolist() if motion_scores else 0
    return avg_hist, motion_score

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
    signature = {}
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
            if current_shot_start < int(cap.get(cv2.CAP_PROP_POS_FRAMES)):
                # Process the last shot
                avg_hist, motion_score = process_shot(shot_frames, current_shot_histograms)
                histograms.append(avg_hist)
                motion_scores.append(motion_score)
                shot_boundaries.append((current_shot_start, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1))
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

        if hist_diff > color_threshold or motion_value > motion_threshold:
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Process the shot that just ended
            avg_hist, motion_score = process_shot(shot_frames, current_shot_histograms)
            histograms.append(avg_hist)  # avg_hist is already a list here
            motion_scores.append(motion_score)  # motion_score is already a single value here
            shot_boundaries.append((current_shot_start, frame_number - 1))
            # Start a new shot
            current_shot_start = frame_number
            current_shot_histograms = []
            shot_frames = []

        # Update the previous frame information
        previous_frame_gray = frame_gray
        previous_hist = current_hist

    cap.release()
    signature = {
        "shotBoundaries": shot_boundaries,
        "histograms": histograms,
        "motionScores": motion_scores
    }

    return signature

# Parameters
color_threshold = 0.5  # Experimentally determined threshold for color change
motion_threshold = 1.5e5  # Experimentally determined threshold for motion change
downsample_percent = 50  # Downsample frames by 50% to reduce computation
signatures = {}
database_path = "Data/Videos"

#Assuming database_path is a directory containing your videos
for video_path in Path(database_path).glob('*.mp4'):
    f = str(video_path)
    print("processing video ", f)
    signatures[f] = detect_shot_boundaries(f, color_threshold, motion_threshold, downsample_percent)


with open("db_signatures.json", 'w') as f:
    json.dump(signatures, f)