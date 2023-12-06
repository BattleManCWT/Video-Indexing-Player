import cv2
import numpy as np
from pathlib import Path
import librosa
from typing import Optional, Tuple, Dict, List
from scipy import signal

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

def detect_shot_histogram(video_path: str, downsample_percent: float) -> Optional[np.ndarray]:
    """
    Detect and compute the average histogram for shots in a video.

    :param video_path: Path to the video file.
    :param downsample_percent: Percentage to downsample each frame for processing.
    :return: The average histogram of the video or None if the video can't be read.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Can't read video file {video_path}.")
        return None

    current_shot_histograms = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_downsampled = downsample_frame(frame, downsample_percent)
        current_hist = calculate_histogram(frame_downsampled)
        current_shot_histograms.append(current_hist)

    cap.release()

    return process_shot(np.array(current_shot_histograms))

def compare_video_signatures(query_histogram: np.ndarray, db_signatures: Dict[str, Dict[str, List[np.ndarray]]]) -> Tuple[str, float]:
    """
    Compare a query histogram against a database of video signatures.

    :param query_histogram: The histogram of the query video.
    :param db_signatures: Database of video signatures.
    :return: Path of the closest video and the similarity score.
    """
    closest_video = None
    min_distance = float('inf')

    for video_path, signature in db_signatures.items():
        for video_histogram in signature['histograms']:
            distance = cv2.compareHist(query_histogram, video_histogram, cv2.HISTCMP_CHISQR)
            if distance < min_distance:
                min_distance = distance
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

def find_offset(within_file, find_file):
    y_within, sr_within = librosa.load(within_file, sr=None)
    y_find, _ = librosa.load(find_file, sr=sr_within)

    c = signal.correlate(y_within, y_find, mode='valid', method='fft')
    peak = np.argmax(c)
    offset = round(peak / sr_within, 2)

    return offset

# def extract_frames(video_path: str, interval: float) -> List[np.ndarray]:
#     """
#     Extract frames from a video at a specified interval.
#
#     :param video_path: Path to the video file.
#     :param interval: Interval at which to extract frames (in seconds).
#     :return: List of extracted frames.
#     """
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frames = []
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         if frame_number % int(fps * interval) == 0:
#             frames.append(frame)
#
#     cap.release()
#     return frames
#
# def find_matching_frame(query_frames: List[np.ndarray], db_frames: List[np.ndarray]) -> Tuple[int, int, float]:
#     """
#     Find the best matching frame from the database for a query video.
#
#     :param query_frames: List of frames from the query video.
#     :param db_frames: List of frames from the database video.
#     :return: Tuple of (query frame index, db frame index, similarity score).
#     """
#     best_match = (None, None, float('inf'))
#     for i, query_frame in enumerate(query_frames):
#         query_hist = calculate_histogram(query_frame)
#         for j, db_frame in enumerate(db_frames):
#             db_hist = calculate_histogram(db_frame)
#             distance = cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_CHISQR)
#             if distance < best_match[2]:
#                 best_match = (i, j, distance)
#
#     return best_match

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
    # query_path = "Data/Queries"
    query_video_path = "Queries/video9_1.mp4"
    query_video_name = query_video_path.split("/")[-1][:-4]

    downsample_percent = 50
    db_signatures = np.load('dbsignatures.npy', allow_pickle=True).item()

    # for video_path in Path(query_path).glob('*.mp4'):
    #     query_histogram = detect_shot_histogram(str(video_path), downsample_percent)
    #     if query_histogram is not None:
    #         closest_video, similarity_score = compare_video_signatures(query_histogram, db_signatures)
    #         print(f"The most similar video to {video_path.name} is {Path(closest_video).name} with a similarity score of {similarity_score}")

    query_histogram = detect_shot_histogram(query_video_path, downsample_percent)
    closest_video, similarity_score = compare_video_signatures(query_histogram, db_signatures)
    print(f"The most similar video to {query_video_path} is {Path(closest_video).name} with a similarity score of {similarity_score}")
    closest_video_name = closest_video.split("/")[-1][:-4]
    offset = find_offset( f"Audios/{closest_video_name}.wav", f"Queries/audios/{query_video_name}.wav")
    offset_minutes = offset // 60
    offset_remainder_seconds = offset % 60
    print(f"Offset: {offset}s" )
    print(f"Offset: {int(offset_minutes)}m {round(offset_remainder_seconds, 2)}s")

    offset = find_offset(f"Audios/{closest_video_name}.wav", f"Queries/audios/{query_video_name}.wav")

    # Assuming you have the FPS value for the database video
    fps = 30  # Replace this with the actual FPS of your database video

    frame_number = calculate_frame_number(offset, fps)
    print(f"Frame number corresponding to the offset: {frame_number}")

    # query_frames = extract_frames("Queries/video9_1.mp4", 1)  # Extract frames every 1 second
    # db_video_path = closest_video  # This is from your existing code
    # db_frames = extract_frames(f"Videos/{closest_video_name}.mp4", 1)
    #
    # query_frame_index, db_frame_index, similarity_score = find_matching_frame(query_frames, db_frames)
    # print(f"Best match at query frame {query_frame_index} and database frame {db_frame_index} with similarity score {similarity_score}")


if __name__ == "__main__":
    main()
