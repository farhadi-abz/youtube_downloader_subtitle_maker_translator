import os
from rich import print


def print_name(video_path):
    srt_filename: str = os.path.splitext(video_path)[0] + ".srt"
    srt_filename = srt_filename.replace("\\", "/")
    print(srt_filename)
