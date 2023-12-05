# main.py

import tkinter as tk
from VideoPlayer import VideoPlayer

def main():
    root = tk.Tk()
    player = VideoPlayer(root, "Video Player", "Videos/video2.mp4", "audios/video2.wav")
    root.mainloop()

if __name__ == "__main__":
    main()
