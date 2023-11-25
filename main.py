import cv2
from videoPlayer import VideoPlayer
import tkinter as tk

# Provide the paths to your video and audio files
default_video_path = "videos/video2.mp4"
default_audio_path = "audios/video2.wav"

# Create the main window
root = tk.Tk()

# Create the VideoPlayer instance with the specified video and audio paths
player = VideoPlayer(root)
player.cap = cv2.VideoCapture(default_video_path)
player.width = int(player.cap.get(3))
player.height = int(player.cap.get(4))
player.audio_path = default_audio_path
player.update_canvas()  # Update canvas with the initial frame

# Run the application
player.run()
