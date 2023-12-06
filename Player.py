import tkinter as tk
from PIL import Image, ImageTk
import threading
import pygame
import wave

class VideoPlayer:
    def __init__(self, root, video_file, audio_file):
        self.root = root
        self.video_file = video_file
        self.audio_file = audio_file
        self.frames = self.load_video(video_file)
        self.current_frame = 0
        self.playing = False
        self.paused = False

        self.label = tk.Label(root)
        self.label.pack()

        play_button = tk.Button(root, text="Play", command=self.play_video)
        play_button.pack()

        pause_button = tk.Button(root, text="Pause", command=self.pause_video)
        pause_button.pack()

        reset_button = tk.Button(root, text="Reset", command=self.reset_video)
        reset_button.pack()

        # Initialize Pygame for audio
        pygame.init()
        pygame.mixer.init()
        self.audio_loaded = False

        # Calculate frame rate
        self.frame_rate = self.calculate_frame_rate()

    def load_video(self, file_path):
        frames = []
        frame_size = 352 * 288 * 3  # 3 bytes per pixel for RGB
        with open(file_path, 'rb') as f:
            while True:
                content = f.read(frame_size)
                if not content:
                    break
                img = Image.frombytes('RGB', (352, 288), content)
                frames.append(ImageTk.PhotoImage(img))
        return frames

    def calculate_frame_rate(self):
        with wave.open(self.audio_file, 'rb') as wav:
            length_in_seconds = wav.getnframes() / float(wav.getframerate())
        return len(self.frames) / length_in_seconds

    def update_frame(self):
        if self.playing and self.current_frame < len(self.frames):
            self.label.configure(image=self.frames[self.current_frame])
            self.current_frame += 1
            self.root.after(int(1000 / self.frame_rate), self.update_frame)

    def play_video(self):
        if not self.playing:
            self.playing = True
            if not self.audio_loaded:
                pygame.mixer.music.load(self.audio_file)
                self.audio_loaded = True
            pygame.mixer.music.play(start=self.current_frame / self.frame_rate)
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
        self.label.configure(image=self.frames[0])
        pygame.mixer.music.stop()
        pygame.mixer.music.load(self.audio_file)  # Reload the audio to reset it

    def get_frame_from_time(self, start_time_seconds):
        return int(start_time_seconds * self.frame_rate)

    def play_from_frame(self, frame_num):
        if 0 <= frame_num < len(self.frames):
            self.current_frame = frame_num
            self.play_video()
        else:
            print("Frame number out of range")

if __name__ == "__main__":
    root = tk.Tk()
    vp = VideoPlayer(root, 'Test_rgb/video2_1.rgb', 'Data/Queries/audios/video2_1.wav')
    start_time_seconds = 5  # Start time in seconds
    frame_to_start = vp.get_frame_from_time(start_time_seconds)
    vp.play_from_frame(frame_to_start)
    root.mainloop()
