import os
import subprocess

def create_rgb_files(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".mp4"):
            input_file = os.path.join(input_folder, file)
            output_file = os.path.join(output_folder, f'{file.split(".")[0]}.rgb')
            subprocess.run(['ffmpeg', '-i', input_file, '-vf', 'format=rgb24', output_file])

def main():
    # original video
    input_folder = "Videos"
    # output
    output_folder = "Videos_rgb"

    os.makedirs(output_folder, exist_ok=True)

    create_rgb_files(input_folder, output_folder)

if __name__ == "__main__":
    main()

