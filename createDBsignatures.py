import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

def downsample_frame(frame: np.ndarray, scale_percent: float) -> np.ndarray:
    """
    Downsample the given frame by the specified scale percentage.

    :param frame: Input frame to downsample.
    :param scale_percent: Scale percentage to downsample the frame.
    :return: Downsampled frame.
    """
    scale = scale_percent / 100
    dimensions = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def calculate_histogram(frame: np.ndarray, bins_per_channel: Tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    """
    Calculate the color histogram for the given frame.

    :param frame: Frame for which to calculate the histogram.
    :param bins_per_channel: Number of bins per channel.
    :return: Flattened histogram of the frame.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, bins_per_channel, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def process_shot(current_shot_histograms: np.ndarray) -> np.ndarray:
    """
    Process the current shot histograms and return the average histogram.

    :param current_shot_histograms: Array of histograms for the current shot.
    :return: Average histogram for the shot.
    """
    if current_shot_histograms.size == 0:
        return np.zeros(8 * 8 * 8)
    return np.mean(current_shot_histograms, axis=0)

def create_database_signatures(video_path: str, color_threshold: float, downsample_percent: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect shot boundaries and calculate histograms for each shot in the video.

    :param video_path: Path to the video file.
    :param color_threshold: Threshold for detecting shot boundaries based on color histogram differences.
    :param downsample_percent: Percentage to downsample each frame for processing.
    :return: Tuple of arrays containing shot boundaries and histograms, or None if the video cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Can't read video file {video_path}.")
        return None

    previous_frame = downsample_frame(frame, downsample_percent)
    previous_hist = calculate_histogram(previous_frame)
    shot_boundaries = []
    histograms = []
    current_shot_histograms = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_downsampled = downsample_frame(frame, downsample_percent)
        current_hist = calculate_histogram(frame_downsampled)
        current_shot_histograms.append(current_hist)

        hist_diff = cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CHISQR)
        if hist_diff > color_threshold:
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            avg_hist = process_shot(np.array(current_shot_histograms))
            histograms.append(avg_hist)
            shot_boundaries.append((frame_number, frame_number))
            current_shot_histograms = []

        previous_hist = current_hist
    
    if(current_shot_histograms):
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        avg_hist = process_shot(np.array(current_shot_histograms))
        histograms.append(avg_hist)
        shot_boundaries.append((frame_number, frame_number))

    cap.release()
    return np.array(shot_boundaries), np.array(histograms)

def main():
    color_threshold = 0.5
    downsample_percent = 50
    signatures: Dict[str, Dict[str, np.ndarray]] = {}
    database_path = "Data/Videos"

    for video_path in Path(database_path).glob('*.mp4'):
        print(f"Processing video {video_path}")
        result = create_database_signatures(str(video_path), color_threshold, downsample_percent)
        if result is not None:
            shot_boundaries, histograms = result
            signatures[str(video_path)] = {"shotBoundaries": shot_boundaries, "histograms": histograms}

    np.save("dbsignatures.npy", signatures)

if __name__ == "__main__":
    main()
