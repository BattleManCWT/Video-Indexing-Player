# main.py
import tkinter as tk
import numpy as np
from pathlib import Path
from Player import VideoPlayer
from compareSignature import create_query_signature, compare_video_signatures, find_offset, calculate_frame_number
from rgbCompare import locate_exact_frame


def main():
    query_video_path = "/Volumes/T7 Shield/Queries/MP4/video7_1.mp4"
    # query_video_path = "Test/video15_3.mp4"
    query_video_name = query_video_path.split("/")[-1][:-4]

    downsample_percent = 50
    color_threshold = 0.5
    db_signatures = np.load('dbsignatures.npy', allow_pickle=True).item()

    query_histogram = create_query_signature(query_video_path, color_threshold, downsample_percent)
    closest_video, similarity_score = compare_video_signatures(query_histogram, db_signatures)
    print(
        f"The most similar video to {query_video_path} is {Path(closest_video).name} with a similarity score of {similarity_score}")
    closest_video_name = closest_video.split("/")[-1][:-4]
    offset = find_offset(f"/Volumes/T7 Shield/Videos/Audios/{closest_video_name}.wav", f"/Volumes/T7 Shield/Queries/Audios/{query_video_name}.wav")
    # offset = find_offset(f"/Volumes/T7 Shield/Videos/Audios/{closest_video_name}.wav", f"Test/{query_video_name}.wav")
    offset_minutes = offset // 60
    offset_remainder_seconds = offset % 60
    print(f"Offset: {offset}s")
    print(f"Offset: {int(offset_minutes)}m {round(offset_remainder_seconds, 2)}s")

    # Calculate the specific frame
    approximate_location = calculate_frame_number(offset, 30)
    frame_to_start = locate_exact_frame(f"/Volumes/T7 Shield/Queries/RGB_Files/{query_video_name}.rgb",
                                        f"/Volumes/T7 Shield/Videos/rgb_files/{closest_video_name}.rgb", approximate_location, 5)
    # frame_to_start = locate_exact_frame(f"Test_rgb/{query_video_name}.rgb",
    #                                     f"/Volumes/T7 Shield/Videos/rgb_files/{closest_video_name}.rgb", approximate_location, 5)
    print(f"Start frame: {frame_to_start}")
    # Initialize the video player
    root = tk.Tk()
    root.title(f"{closest_video_name}")
    player = VideoPlayer(root, f"/Volumes/T7 Shield/Videos/rgb_files/{closest_video_name}.rgb", f"/Volumes/T7 Shield/Videos/Audios/{closest_video_name}.wav",
                         frame_to_start)
    player.play_from_frame(frame_to_start)
    player.pause_video()
    root.mainloop()


if __name__ == "__main__":
    main()
