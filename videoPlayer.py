# VideoPlayer.py

import ffmpeg
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pygame

class VideoPlayer:
    def __init__(self, window, window_title, video_path, audio_path=None):
        self.window = window
        self.window.title(window_title)
        self.video_path = video_path
        self.audio_path = audio_path

        # Initialize Pygame for audio playback
        if self.audio_path:
            pygame.init()
            pygame.mixer.init()
            self.audio = pygame.mixer.Sound(audio_path)

        # Open the video file with ffmpeg
        self.process = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )

        # Get video properties
        self.vid = cv2.VideoCapture(video_path)
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Play button
        self.btn_play = tk.Button(window, text="Play", command=self.play)
        self.btn_play.pack(side=tk.LEFT)

        # Pause button
        self.btn_pause = tk.Button(window, text="Pause", command=self.pause)
        self.btn_pause.pack(side=tk.LEFT)

        # Reset button
        self.btn_reset = tk.Button(window, text="Reset", command=self.reset)
        self.btn_reset.pack(side=tk.LEFT)

        self.running = False

    def play(self):
        if not self.running:
            self.running = True
            if self.audio_path:
                pygame.mixer.unpause()  # Unpause the audio if it was paused
                if not pygame.mixer.get_busy():  # Play only if the audio is not already playing
                    pygame.mixer.Sound.play(self.audio)
            self.read_frame()

    def pause(self):
        self.running = False
        if self.audio_path:
            pygame.mixer.pause()  # Pause the audio

    def reset(self):
        self.running = False
        if self.audio_path:
            pygame.mixer.Sound.stop(self.audio)

        # Kill the existing FFmpeg process
        if self.process is not None:
            self.process.kill()

        # Start a new FFmpeg process
        self.process = (
            ffmpeg
            .input(self.video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )

        # Immediately read and display the first frame
        in_bytes = self.process.stdout.read(self.width * self.height * 3)
        if in_bytes:
            frame = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([self.height, self.width, 3])
            )
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def read_frame(self):
        if self.running:
            in_bytes = self.process.stdout.read(self.width * self.height * 3)
            if not in_bytes:
                self.running = False
                return
            frame = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([self.height, self.width, 3])
            )
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.window.after(15, self.read_frame)

    def __del__(self):
        if self.process is not None:
            self.process.kill()
        if self.vid.isOpened():
            self.vid.release()
        if self.audio_path:
            pygame.quit()

