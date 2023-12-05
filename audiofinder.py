import librosa
import numpy as np
from scipy import signal


def find_offset(within_file, find_file, window):
    y_within, sr_within = librosa.load(within_file, sr=None)
    y_find, _ = librosa.load(find_file, sr=sr_within)

    c = signal.correlate(y_within, y_find[:sr_within*window], mode='valid', method='fft')
    peak = np.argmax(c)
    offset = round(peak / sr_within, 2)

    return offset


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--find-offset-of', metavar='audio file', type=str, help='Find the offset of file')
    # parser.add_argument('--within', metavar='audio file', type=str, help='Within file')
    # parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of a target audio')
    # args = parser.parse_args()
    offset = find_offset( "Data/Audios/video3.wav", "Data/Queries/audios/video3_1.wav", 900)
    offset_minutes = offset // 60
    offset_remainder_seconds = offset % 60
    print(f"Offset: {offset}s" )
    print(f"Offset: {int(offset_minutes)}m {round(offset_remainder_seconds, 2)}s")
    


if __name__ == '__main__':
    main()