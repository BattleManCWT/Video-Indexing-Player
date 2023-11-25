import json
import numpy as np
import cv2
from scipy.spatial import distance

def downsample_frame(frame, scale_percent):
    """Reduce the size of the frame to the specified percentage."""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def calculate_histogram(frame):
    """Calculate a normalized color histogram for the given frame."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def calculate_motion(previous_frame, current_frame):
    """Calculate the magnitude of motion between two frames using optical flow."""
    flow = cv2.calcOpticalFlowFarneback(
        previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.sum(magnitude)

def calculate_optical_flows(frames):
    """Calculate motion magnitudes using optical flows between a list of frames."""
    motion_magnitudes = []
    prev_frame_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        next_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitudes.append(np.mean(magnitude))
        prev_frame_gray = next_frame_gray

    return motion_magnitudes

def process_shot(shot_frames, current_shot_histograms):
    """Process the accumulated frames of a shot to calculate histogram & motion."""
    if current_shot_histograms:
        avg_hist = np.mean(np.array(current_shot_histograms), axis=0).tolist()
    else:
        avg_hist = [0] * (8 * 8 * 8)

    motion_scores = calculate_optical_flows(shot_frames)
    if motion_scores:
        motion_score = np.mean(motion_scores).tolist()
    else:
        motion_score = 0

    return avg_hist, motion_score

def detect_shot_boundaries(video_path, color_threshold, motion_threshold,
                           downsample_percent):
    """Detect shot boundaries in a video and return their signatures."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't read video file.")
        return [], []

    previous_frame = downsample_frame(frame, downsample_percent)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_hist = calculate_histogram(previous_frame)
    shot_boundaries, histograms, motion_scores = [], [], []
    shot_frames, current_shot_histograms = [], []
    current_shot_start = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if current_shot_start < int(cap.get(cv2.CAP_PROP_POS_FRAMES)):
                avg_hist, motion_score = process_shot(shot_frames,
                                                      current_shot_histograms)
                histograms.append(avg_hist)
                motion_scores.append(motion_score)
                end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                shot_boundaries.append((current_shot_start, end_frame))
            break

        frame_downsampled = downsample_frame(frame, downsample_percent)
        shot_frames.append(frame_downsampled)
        frame_gray = cv2.cvtColor(frame_downsampled, cv2.COLOR_BGR2GRAY)
        current_hist = calculate_histogram(frame_downsampled)
        current_shot_histograms.append(current_hist)

        hist_diff = cv2.compareHist(previous_hist, current_hist,
                                    cv2.HISTCMP_BHATTACHARYYA)
        motion_value = calculate_motion(previous_frame_gray, frame_gray)

        if hist_diff > color_threshold or motion_value > motion_threshold:
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            avg_hist, motion_score = process_shot(shot_frames,
                                                  current_shot_histograms)
            histograms.append(avg_hist)
            motion_scores.append(motion_score)
            shot_boundaries.append((current_shot_start, frame_number - 1))
            current_shot_start = frame_number
            shot_frames, current_shot_histograms = [], []

        previous_frame_gray, previous_hist = frame_gray, current_hist

    cap.release()
    return {"shotBoundaries": shot_boundaries, "histograms": histograms,
            "motionScores": motion_scores}

def compare_signatures(query_signature, db_signatures):
    """Compare video signatures to find the best match."""
    match_scores = {}

    for video_path, video_signature in db_signatures.items():
        shot_scores = [
            min(
                [distance.euclidean(q_hist, db_hist) + abs(q_motion - db_motion)
                 for db_shot_boundary, db_hist, db_motion in zip(
                     video_signature['shotBoundaries'],
                     video_signature['histograms'],
                     video_signature['motionScores']
                 )]
            ) if query_signature['shotBoundaries'] else float('inf')
            for q_shot_boundary, q_hist, q_motion in zip(
                query_signature['shotBoundaries'],
                query_signature['histograms'],
                query_signature['motionScores']
            )
        ]

        avg_shot_score = (sum(shot_scores) / len(shot_scores)
                          if shot_scores else float('inf'))
        match_scores[video_path] = avg_shot_score

    best_match_video = min(match_scores, key=match_scores.get)
    return best_match_video, match_scores[best_match_video]

# Example usage:
color_threshold = 0.5
motion_threshold = 1.5e5
downsample_percent = 50

query_path = "Data/Queries/video2_1.mp4"
query_signature = detect_shot_boundaries(query_path, color_threshold,
                                         motion_threshold, downsample_percent)

with open('signatures.json') as f:
    data = json.load(f)

best_match, score = compare_signatures(query_signature, data)
print(f"The best match is {best_match} with a score of {score}")
