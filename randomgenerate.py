from moviepy.editor import VideoFileClip
import random

def generate_random_clips(video_path, number_of_clips=5, min_duration=10, max_duration=40):
    # 读取视频
    video = VideoFileClip(video_path)
    duration = video.duration

    clips = []

    for _ in range(number_of_clips):
        # 随机选择开始时间
        start = random.uniform(0, duration - max_duration)
        # 随机选择片段时长
        clip_duration = random.uniform(min_duration, max_duration)
        end = start + clip_duration

        # 截取片段
        clip = video.subclip(start, end)
        clips.append(clip)

        # 导出片段
        clip.write_videofile(f"clip_{start:.2f}_{end:.2f}.mp4")

# 使用示例
generate_random_clips("Data/Videos/video12.mp4")
