import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

def downsample_frame(frame: np.ndarray, scale_percent: float) -> np.ndarray:
    """
    Downsample the given frame by the specified scale percentage.
    """
    scale = scale_percent / 100
    dimensions = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def calculate_histogram(frame: np.ndarray, bins_per_channel: Tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    """
    Calculate the color histogram for the given frame.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, bins_per_channel, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def process_shot(current_shot_histograms: np.ndarray) -> np.ndarray:
    """
    Process the current shot histograms and return the average histogram.
    """
    if current_shot_histograms.size == 0:
        return np.zeros(8 * 8 * 8)
    return np.mean(current_shot_histograms, axis=0)



def calculate_motion_score(previous_frame: np.ndarray, current_frame: np.ndarray) -> float:
    """
    Calculate the motion score between two frames.
    """
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    score = np.sum(frame_diff) / frame_diff.size
    return score

def create_database_signatures(video_path: str, color_threshold: float, motion_threshold: float, downsample_percent: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Detect shot boundaries and calculate histograms and motion scores for each shot in the video.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Can't read video file {video_path}.")
        return None

    previous_frame = downsample_frame(frame, downsample_percent)
    previous_hist = calculate_histogram(previous_frame, bins_per_channel=(8,8,8))
    shot_boundaries = []
    histograms = []
    motion_scores = []
    current_shot_histograms = []
    current_shot_motion_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_downsampled = downsample_frame(frame, downsample_percent)
        current_hist = calculate_histogram(frame_downsampled, bins_per_channel=(8,8,8))
        motion_score = calculate_motion_score(previous_frame, frame_downsampled)

        current_shot_histograms.append(current_hist)
        current_shot_motion_scores.append(motion_score)


        hist_diff = cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CHISQR)

        if hist_diff > color_threshold or motion_score > motion_threshold:
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            avg_hist = process_shot(np.array(current_shot_histograms))
            avg_motion_score = np.mean(current_shot_motion_scores)
            histograms.append(avg_hist)
            motion_scores.append(avg_motion_score)
            shot_boundaries.append((frame_number, frame_number))
            current_shot_histograms = []
            current_shot_motion_scores = []

        previous_hist = current_hist
        previous_frame = frame_downsampled

    cap.release()
    return np.array(shot_boundaries), np.array(histograms), np.array(motion_scores)

def main():
    """
    Main function to process all videos in the given path.
    """
    color_threshold = 0.6
    motion_threshold = 60.0
    downsample_percent = 70
    signatures: Dict[str, Dict[str, np.ndarray]] = {}
    database_path = "Data/Videos"

    for video_path in Path(database_path).glob('*.mp4'):
        print(f"Processing video {video_path}")
        result = create_database_signatures(str(video_path), color_threshold, motion_threshold, downsample_percent)
        if result is not None:
            shot_boundaries, histograms, motion_scores = result
            signatures[str(video_path)] = {
                "shotBoundaries": shot_boundaries, 
                "histograms": histograms, 
                "motionScores": motion_scores
            }

    np.save("dbsignatures3.npy", signatures)

if __name__ == "__main__":
    main()
