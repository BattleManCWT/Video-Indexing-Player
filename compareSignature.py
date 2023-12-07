import cv2
import numpy as np
from pathlib import Path
import librosa
from typing import Optional, Tuple, Dict, List
from scipy import signal
from statistics import mean

def downsample_frame(frame: np.ndarray, scale_percent: float) -> np.ndarray:
    """
    Downsample a frame by a specified scale percentage.

    :param frame: The frame to downsample.
    :param scale_percent: The percentage to scale down by.
    :return: Downsampled frame.
    """
    scale = scale_percent / 100
    dimensions = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def calculate_histogram(frame: np.ndarray, bins_per_channel: Tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    """
    Calculate the color histogram for a given frame.

    :param frame: The frame to calculate the histogram for.
    :param bins_per_channel: The number of bins per channel.
    :return: The flattened histogram.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, bins_per_channel, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def process_shot(current_shot_histograms: np.ndarray) -> np.ndarray:
    """
    Compute the average histogram of the current shot.

    :param current_shot_histograms: Array of histograms for the current shot.
    :return: Average histogram.
    """
    if current_shot_histograms.size == 0:
        return np.zeros(8 * 8 * 8)
    return np.mean(current_shot_histograms, axis=0)

def create_query_signature(video_path: str, color_threshold: float, downsample_percent: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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
            avg_hist = process_shot(np.array(current_shot_histograms))
            histograms.append(avg_hist)
            current_shot_histograms = []

        previous_hist = current_hist

    if current_shot_histograms:
        avg_hist = process_shot(np.array(current_shot_histograms))
        histograms.append(avg_hist)

    cap.release()
    return np.array(histograms)

def compare_video_signatures(query_histograms, db_signatures: Dict[str, Dict[str, List[np.ndarray]]]) -> Tuple[str, float]:
    """
    Compare a query histogram against a database of video signatures.

    :param query_histograms: The histogram of the query video.
    :param db_signatures: Database of video signatures.
    :return: Path of the closest video and the similarity score.
    """
    closest_video = None
    min_distance = float('inf')
    query_len = len(query_histograms)

    for video_path, signature in db_signatures.items():
        hist_distances = []
        for h in query_histograms:
            submindist = float('inf')
            for video_histogram in signature['histograms']:
               
                distance = cv2.compareHist(h, video_histogram, cv2.HISTCMP_BHATTACHARYYA)
                if distance < submindist:
                    submindist = distance
            hist_distances.append(submindist)

        avg_distance = mean(hist_distances)
        if avg_distance < min_distance:
            min_distance = avg_distance
            closest_video = video_path

    return closest_video, min_distance

def frame_to_time(frame_number: int, fps: int = 30) -> str:
    """
    Convert a frame number to a timestamp.

    :param frame_number: The frame number.
    :param fps: Frames per second of the video.
    :return: Timestamp string in 'HH:MM:SS' format.
    """
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def find_offset(within_file, find_file, downsample_rate=22050, y_find_duration=None):
    # Load with a lower sample rate to reduce data size
    y_within, sr_within = librosa.load(within_file, sr=downsample_rate)

    # Load y_find with the same sample rate, and optionally only a portion of it
    y_find, _ = librosa.load(find_file, sr=downsample_rate, duration=y_find_duration)

    # Perform cross-correlation
    c = signal.correlate(y_within, y_find, mode='valid', method='fft')
    peak = np.argmax(c)
    offset = round(peak / sr_within, 2)

    return offset

def calculate_frame_number(offset_seconds: float, fps: int) -> int:
    """
    Calculate the frame number based on the offset time and FPS.
    :param offset_seconds: Offset time in seconds.
    :param fps: Frames per second of the video.
    :return: Frame number corresponding to the offset time.
    """
    frame_number = int(offset_seconds * fps)
    return frame_number

def main():
    query_path = "Data/Queries/video9_1.mp4"
    query_video_name = query_path.split("/")[-1][:-4]
    video_name = query_video_name

    downsample_percent = 50
    color_threshold = 0.5
    db_signatures = np.load('dbsignatures.npy', allow_pickle=True).item()

    query_histograms = create_query_signature(query_path,color_threshold, downsample_percent)
    closest_video, similarity_score = compare_video_signatures(query_histograms, db_signatures)
    print(f"The most similar video to {query_video_name} is {Path(closest_video).name}")
    closest_video_name = closest_video.split("/")[-1][:-4]
    offset = find_offset( f"Data/Audios/{closest_video_name}.wav", f"Data/Queries/audios/{query_video_name}.wav")

    offset_minutes = offset // 60
    offset_remainder_seconds = offset % 60
    print(f"Offset: {offset}s" )
    print(f"Offset: {int(offset_minutes)}m {round(offset_remainder_seconds, 2)}s")

if __name__ == "__main__":
    main()
