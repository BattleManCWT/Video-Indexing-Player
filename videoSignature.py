import cv2
import numpy as np
import hashlib
from pathlib import Path
from sklearn.cluster import KMeans
import json


def extract_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    while cap.isOpened():
        # Read frame by frame
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            break
        
        frames.append(frame)
        
    # Release the video capture object
    cap.release()
    
    return frames

import cv2
import numpy as np

def calculate_edge_change_ratio(frame1, frame2):
    """
    Calculate the Edge Change Ratio (ECR) between two frames.

    Parameters:
    - frame1: numpy.ndarray, the first frame
    - frame2: numpy.ndarray, the second frame
    
    Returns:
    - ecr: float, the edge change ratio between frame1 and frame2
    """
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detector
    edges1 = cv2.Canny(gray1, 100, 200)
    edges2 = cv2.Canny(gray2, 100, 200)
    
    # Compute the difference between the edge images
    diff = cv2.absdiff(edges1, edges2)
    
    # Count the edge pixels in both frames and the difference image
    edge_count1 = np.sum(edges1 > 0)
    edge_count2 = np.sum(edges2 > 0)
    diff_edge_count = np.sum(diff > 0)
    
    # Calculate the Edge Change Ratio
    ecr = diff_edge_count / (min(edge_count1, edge_count2) + 1e-10)  # Add small value to avoid division by zero
    
    return ecr

def detect_shot_boundaries(frames, ecr_threshold=0.4):
    """
    Detects shot boundaries in a list of frames using the Edge Change Ratio (ECR).

    Parameters:
    - frames: list of numpy.ndarray, the frames extracted from the video
    - ecr_threshold: float, the threshold for the ECR to consider a shot boundary
    
    Returns:
    - boundaries: list of int, the indices of frames where shot boundaries occur
    """
    boundaries = []

    for i in range(len(frames) - 1):
        ecr = calculate_edge_change_ratio(frames[i], frames[i + 1])
        
        if ecr > ecr_threshold:
            boundaries.append(i + 1)  # The boundary is at the start of the new shot
    
    return boundaries

def compute_color_histograms(frames, bins=256):
    histograms = []

    for frame in frames:
        # Initialize histogram for the frame
        hist_frame = np.zeros((bins*3,))

        # Calculate histograms for each color channel
        for i, channel_col in enumerate(cv2.split(frame)):
            hist = cv2.calcHist([channel_col], [0], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_frame[i*bins:(i+1)*bins] = hist  # Append to the frame's histogram

        # Append the histogram for the frame to the list
        histograms.append(hist_frame)

    return histograms


def compute_motion_vectors(frames):
    """
    Computes the motion vectors between successive frames using optical flow.

    Parameters:
    - frames: list of numpy.ndarray, the frames extracted from the video
    
    Returns:
    - motion_vectors: list of numpy.ndarray, the motion vectors for each pair of frames
    """
    motion_vectors = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow using Farneback's method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Append the flow data (motion vectors) for the frame to the list
        motion_vectors.append(flow)
        prev_gray = gray

    return motion_vectors



def create_signature(shot_boundaries, color_histograms, motion_vectors, normalize=True):
    """
    Creates a digital signature from shot boundaries, color histograms, and motion vectors.

    Parameters:
    - shot_boundaries: list of int, indices of frames where shot boundaries occur
    - color_histograms: list of numpy.ndarray, color histograms for each frame
    - motion_vectors: list of numpy.ndarray, motion vectors for each pair of frames
    - normalize: bool, flag to normalize the features before concatenation

    Returns:
    - signature: numpy.ndarray, the digital signature of the video
    """
    # Start with temporal information of shot boundaries as the structural part of the signature
    # Here, we use the relative positions of shot boundaries as a feature
    shot_boundary_features = np.diff([0] + shot_boundaries + [len(color_histograms)])

    # Normalize the shot boundary features if needed
    if normalize:
        shot_boundary_features = shot_boundary_features / np.sum(shot_boundary_features)

    # For color histograms, we can summarize the color distribution across all frames
    mean_color_hist = np.mean(color_histograms, axis=0)
    var_color_hist = np.var(color_histograms, axis=0)

    # Normalize the color features if needed
    if normalize:
        mean_color_hist = mean_color_hist / np.linalg.norm(mean_color_hist)
        var_color_hist = var_color_hist / np.linalg.norm(var_color_hist)

    # For motion vectors, summarize the motion information
    motion_magnitude = np.array([np.linalg.norm(mv) for mv in motion_vectors])
    mean_motion = np.mean(motion_magnitude)
    var_motion = np.var(motion_magnitude)

    # Normalize the motion features if needed
    if normalize:
        mean_motion = mean_motion / np.linalg.norm(motion_magnitude)
        var_motion = var_motion / np.linalg.norm(motion_magnitude)

    # Combine all features into a single signature
    signature = np.concatenate([
        shot_boundary_features,
        mean_color_hist,
        var_color_hist,
        [mean_motion],
        [var_motion]
    ])

    return signature


def hash_signature(signature):
    """
    Hashes the digital signature of a video to create a compact and unique representation.

    Parameters:
    - signature: numpy.ndarray, the digital signature of the video

    Returns:
    - hash_value: str, the hashed value of the digital signature
    """
    # First, we need to ensure the signature is in a byte format
    signature_bytes = signature.tobytes()

    # Create a hash object
    hash_object = hashlib.sha256()

    # Update the hash object with the signature
    hash_object.update(signature_bytes)

    # Get the hexadecimal digest of the signature
    hash_value = hash_object.hexdigest()

    return hash_value

# This function call is just for demonstration and will not execute in this environment
# Assume we have a digital signature from the create_signature function
# hashed_video_signature = hash_signature(video_signature)


def compare_signatures(signature1, signature2):
    """
    Compares two hashed video signatures to determine their similarity.

    Parameters:
    - signature1: str, the hashed value of the first digital signature
    - signature2: str, the hashed value of the second digital signature

    Returns:
    - similarity: float, the similarity score between the two signatures
    """
    # Convert the hex digests into binary arrays
    binary1 = bytearray.fromhex(signature1)
    binary2 = bytearray.fromhex(signature2)

    # Count the number of differing bits (Hamming distance)
    hamming_distance = sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(binary1, binary2))

    # Normalize the Hamming distance to get a similarity score between 0 and 1
    max_distance = 8 * max(len(binary1), len(binary2))  # 8 bits per byte
    similarity = 1 - (hamming_distance / max_distance)

    return similarity

def outputSignature(filename, video_name, signature, mode):

    output = {video_name: signature.tolist()}

    with open(filename, mode) as outfile:
        json.dump(output, outfile)


# Example usage:

query_video = 'Data/Queries/video2_1.mp4'
# Extract frames from the video
frames = extract_frames(query_video)

# Detect shot boundaries
shot_boundaries = detect_shot_boundaries(frames)

# Compute color histograms for each frame
color_histograms = compute_color_histograms(frames)

# Compute motion vectors for each frame
motion_vectors = compute_motion_vectors(frames)

# Create a digital signature from the descriptors
signature = create_signature(shot_boundaries, color_histograms, motion_vectors)

# Hash the digital signature
hashed_signature = hash_signature(signature)

outputSignature("querySignature.json", query_video.split("/")[-1], signature, "w")

# To compare against another signature
database_signatures = {}
database_path = "database"
highest_similarity = -np.inf 
for video_file in Path(database_path).iterdir():
    print("Currently creating digital signature for {}...".format(video_file))
    dbvframes = extract_frames(str(video_file))
    print("Frame Extraction done!")
    dbvshot_boundaries = detect_shot_boundaries(dbvframes)
    print("Shot Boundaries done!")
    dbvcolor_histograms = compute_color_histograms(dbvframes)
    print("Histrogram done!")
    dbvmotion_vectors = compute_motion_vectors(dbvframes)
    print("Motion vectors done!")
    dbvsignature = create_signature(dbvshot_boundaries, dbvcolor_histograms, dbvmotion_vectors)
    print("Signatrue Created")
    dbvhashed_signature = hash_signature(signature)
    print("Finished!")
    print()

    database_signatures[video_file] = dbvsignature

    similarity_score = compare_signatures(hashed_signature, dbvhashed_signature)
    

    if similarity_score > highest_similarity:
        highest_similarity = similarity_score
        best_match = video_file

with open('all_video_signatures.json', 'w') as json_file:
    json.dump(dbvhashed_signature, json_file, indent=4)

print(f"The best match is {video_file} with a similarity of {highest_similarity}")
