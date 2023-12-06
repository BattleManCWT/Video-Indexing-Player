from moviepy.editor import VideoFileClip, AudioFileClip
import random

def generate_random_clips(video_path, video_audio_file, number_of_clips=3, min_duration=5, max_duration=40):
    # Load the video and its audio
    video = VideoFileClip(video_path)
    audio = AudioFileClip(video_audio_file)
    video_name = video_path.split("/")[-1][:4]
    duration = min(video.duration, audio.duration)  # Ensure both video and audio have the same maximum duration

    clips = []

    for _ in range(number_of_clips):
        # Randomly select start time
        start = random.uniform(0, duration - max_duration)
        # Randomly select clip duration
        clip_duration = random.uniform(min_duration, max_duration)
        end = start + clip_duration

        # Create video and audio subclips
        video_clip = video.subclip(start, end)
        audio_clip = audio.subclip(start, end)

        # Set the audio of the video clip
        video_clip = video_clip.set_audio(audio_clip)

        clips.append(video_clip)

        # Export the video and audio clips
        output_filename = f"{video_name}_{start:.2f}_{end:.2f}"
        video_clip.write_videofile(f"{output_filename}.mp4")
        audio_clip.write_audiofile(f"{output_filename}.wav")

# Example usage
generate_random_clips("Data/Videos/video12.mp4", "Data/Audios/video3.wav")
