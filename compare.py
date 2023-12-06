import cv2
import numpy as np
from pathlib import Path
import librosa
from typing import Optional, Tuple, Dict, List
from scipy import signal

def downsample_frame(frame: np.ndarray, scale_percent: float) -> np.ndarray:
    """
    Downsample the given frame by the specified scale percentage.
    """
    scale = scale_percent / 100
    dimensions = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def normalize_motion_score(score, min_score = 0, max_score = 255):
    return (score - min_score) / (max_score - min_score)

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

def create_query_signatures(video_path: str, downsample_percent: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate histograms and motion scores for the entire video, treating it as one whole shot.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Can't read video file {video_path}.")
        return None

    previous_frame = downsample_frame(frame, downsample_percent)
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

        previous_frame = frame_downsampled

    # Compute the average histogram and motion score for the entire video

    avg_hist = process_shot(np.array(current_shot_histograms))
    avg_motion_score = np.mean(current_shot_motion_scores)
    histograms.append(avg_hist)
    motion_scores.append(avg_motion_score)

    cap.release()
    return avg_hist, avg_motion_score

def compare_video_signatures(query_signature, db_signatures):
    """
    Compare a query signature against a database of video signatures.

    :param query_signature: The histogram and motion score of the query video.
    :param db_signatures: Database of video signatures.
    :return: Path of the closest video and the similarity score.
    """
    closest_video = None
    min_distance = float('inf')

    query_histogram, query_motion_score = query_signature
    query_histogram /= np.linalg.norm(query_histogram)
    query_motion_score = normalize_motion_score(query_motion_score)

    for video_path, signature in db_signatures.items():
        db_histograms = signature['histograms']
        db_motion_scores = signature['motionScores']

        for i in range(len(db_histograms)):
            video_histogram = db_histograms[i]
            video_motion_score = db_motion_scores[i]
            video_histogram /= np.linalg.norm(video_histogram)
            video_motion_score = normalize_motion_score(video_motion_score)
            hist_distance = cv2.compareHist(query_histogram, video_histogram, cv2.HISTCMP_BHATTACHARYYA)
            
            motion_distance = abs(query_motion_score - video_motion_score)
            total_distance =  0.4 * hist_distance + 0.6 * motion_distance

            if total_distance < min_distance:
                min_distance = total_distance
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

# def find_offset(within_file, find_file):
#     y_within, sr_within = librosa.load(within_file, sr=None)
#     y_find, _ = librosa.load(find_file, sr=sr_within)

#     c = signal.correlate(y_within, y_find, mode='valid', method='fft')
#     peak = np.argmax(c)
#     offset = round(peak / sr_within, 2)

#     return offset

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


def main():

    for video_path in Path("test").glob('*.mp4'):
        query_video_name = str(video_path).split("/")[-1][:-4]

        downsample_percent = 70
        db_signatures = np.load('dbsignatures2.npy', allow_pickle=True).item()

        query_histogram = create_query_signatures(str(video_path), downsample_percent)
        closest_video, similarity_score = compare_video_signatures(query_histogram, db_signatures)
        print(f"The most similar video to {query_video_name} is {Path(closest_video).name}")
        closest_video_name = closest_video.split("/")[-1][:-4]
        offset = find_offset( f"Data/Audios/{closest_video_name}.wav", f"test/{query_video_name}.wav")
        offset_minutes = offset // 60

if __name__ == "__main__":
    main()
