import cv2
import tkinter as tk
from tkinter import filedialog
import pygame

class VideoPlayer:
    def __init__(self, master):
        self.master = master
        self.master.title("Video Player")

        self.cap = None
        self.width, self.height = 0, 0

        self.audio_path = ""

        self.canvas = tk.Canvas(master)
        self.canvas.pack()

        self.btn_reset = tk.Button(master, text="Reset", command=self.reset_video)
        self.btn_play = tk.Button(master, text="Play", command=self.play_video)
        self.btn_pause = tk.Button(master, text="Pause", command=self.pause_video)
        self.btn_open = tk.Button(master, text="Open Video", command=self.open_video)
        self.btn_reset.pack(side=tk.LEFT)
        self.btn_play.pack(side=tk.LEFT)
        self.btn_pause.pack(side=tk.LEFT)
        self.btn_open.pack(side=tk.LEFT)

        self.is_playing = False
        self.update()

    def reset_video(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.update_canvas()

    def play_video(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        self.is_playing = True
        pygame.mixer.music.unpause()
        self.play_audio()

    def pause_video(self):
        self.is_playing = False
        pygame.mixer.music.pause()

    def open_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.width = int(self.cap.get(3))
            self.height = int(self.cap.get(4))
            self.audio_path = ""  # Reset the audio path when a new video is opened
            self.update_canvas()

            audio_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav")])
            if audio_path:
                self.audio_path = audio_path

    def play_audio(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(self.audio_path)
            pygame.mixer.music.play(start=pygame.mixer.music.get_pos())

    def update_canvas(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = tk.PhotoImage(data=self.convert_frame(frame))
                self.canvas.config(width=self.width, height=self.height)
                self.canvas.img_tk = img
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)

    def convert_frame(self, frame):
        return cv2.imencode('.PPM', frame)[1].tobytes()

    def update(self):
        if self.is_playing:
            self.update_canvas()

        self.master.after(30, self.update)

    def run(self):
        self.master.mainloop()
