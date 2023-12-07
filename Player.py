import tkinter as tk
from PIL import Image, ImageTk
import threading
import pygame
import numpy as np
import os
import wave

class VideoPlayer:
    def __init__(self, root, video_file, audio_file, start_frame=0):
        self.root = root
        self.video_file = video_file
        self.audio_file = audio_file
        self.width = 352  # Set the width of your video
        self.height = 288  # Set the height of your video
        self.frame_size = self.width * self.height * 3
        self.frame_count = self.get_frame_count()
        self.current_frame = start_frame
        self.playing = False
        self.paused = False

        self.label = tk.Label(root)
        self.label.pack()

        button_frame = tk.Frame(root)
        button_frame.pack()

        play_button = tk.Button(button_frame, text="Play", command=self.play_video)
        play_button.pack(side=tk.LEFT)

        pause_button = tk.Button(button_frame, text="Pause", command=self.pause_video)
        pause_button.pack(side=tk.LEFT)

        reset_button = tk.Button(button_frame, text="Reset", command=self.reset_video)
        reset_button.pack(side=tk.LEFT)

        # Initialize Pygame for audio
        pygame.init()
        pygame.mixer.init()
        self.audio_loaded = False

        # Display the initial frame
        self.display_frame(start_frame)

    def get_frame_count(self):
        file_size = os.path.getsize(self.video_file)
        return file_size // self.frame_size

    def read_frame(self, frame_num):
        with open(self.video_file, 'rb') as f:
            f.seek(frame_num * self.frame_size)
            frame_data = np.frombuffer(f.read(self.frame_size), dtype=np.uint8)
            frame_data = frame_data.reshape((self.height, self.width, 3))
            return Image.fromarray(frame_data)

    def display_frame(self, frame_num):
        if 0 <= frame_num < self.frame_count:
            frame_image = ImageTk.PhotoImage(self.read_frame(frame_num))
            self.label.configure(image=frame_image)
            self.label.image = frame_image  # Keep reference

    def calculate_frame_rate(self):
        with wave.open(self.audio_file, 'rb') as wav:
            length_in_seconds = wav.getnframes() / float(wav.getframerate())
        return self.frame_count / length_in_seconds

    def update_frame(self):
        if self.playing and self.current_frame < self.frame_count:
            self.display_frame(self.current_frame)
            self.current_frame += 1
            self.root.after(int(1000 / self.calculate_frame_rate()), self.update_frame)

    def play_video(self):
        if not self.playing:
            self.playing = True
            if not self.audio_loaded:
                pygame.mixer.music.load(self.audio_file)
                self.audio_loaded = True
                audio_start_pos = self.current_frame / self.calculate_frame_rate()
                pygame.mixer.music.play(start=audio_start_pos)
            else:
                if self.paused:
                    pygame.mixer.music.unpause()  # Unpause if paused
                else:
                    audio_start_pos = self.current_frame / self.calculate_frame_rate()
                    pygame.mixer.music.play(start=audio_start_pos)

            threading.Thread(target=self.update_frame).start()
        self.paused = False


    def pause_video(self):
        if self.playing:
            self.playing = False
            self.paused = True
            pygame.mixer.music.pause()

    def reset_video(self):
        self.playing = False
        self.paused = False
        self.current_frame = 0
        self.display_frame(self.current_frame)
        pygame.mixer.music.stop()
        pygame.mixer.music.load(self.audio_file)  # Reload the audio to reset it

    def play_from_frame(self, frame_num):
        if 0 <= frame_num < self.frame_count:
            self.current_frame = frame_num
            if self.playing:
                self.pause_video()  # Pause if already playing

            # Calculate the audio start position in seconds
            audio_start_pos = self.current_frame / self.calculate_frame_rate()

            if not self.audio_loaded:
                pygame.mixer.music.load(self.audio_file)
                self.audio_loaded = True

            pygame.mixer.music.play(start=audio_start_pos)
            threading.Thread(target=self.update_frame).start()
            self.playing = True
            self.paused = False
        else:
            print("Frame number out of range")

if __name__ == "__main__":
    root = tk.Tk()
    vp = VideoPlayer(root, 'path_to_video_file.rgb', 'path_to_audio_file.wav', start_frame=100)
    root.mainloop()
