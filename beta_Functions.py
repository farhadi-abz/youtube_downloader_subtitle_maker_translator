import moviepy
from moviepy.video.tools.subtitles import SubtitlesClip

import os
import yt_dlp
from rich import print
import whisper
import torch
import ollama


print(moviepy.config.check())


def youtube_download(
    url: str,
    output_folder_path: str = "./Movies/",
    quality: str = "best",
):
    if quality == "bestaudio" or quality == "worstaudio":
        file_format = "mp3"
    else:
        file_format = "mp4"
    ydl_opts = {
        "format": f"{quality}",
        "outtmpl": os.path.join(output_folder_path, "%(title)s.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": f"{file_format}",
            }
        ],
    }

    final_file_path = ""

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)
        final_file_path = os.path.splitext(filename)[0] + f".{file_format}"

    print(
        f" ---> The video is downloaded from [bold red]youtube[/bold red] and saved in : [bold blue]{final_file_path}[/bold blue]"
    )
    return final_file_path


def audio_extractor(file_path: str):
    clip = moviepy.VideoFileClip(filename=file_path)
    extracted_audio_file_path = os.path.splitext(file_path)[0] + ".mp3"
    if os.path.exists(extracted_audio_file_path):
        print(
            f" ---> The file [bold blue]{extracted_audio_file_path}[/bold blue] is already exist"
        )
    else:
        print(extracted_audio_file_path)
        if clip.audio is not None:
            clip.audio.write_audiofile(extracted_audio_file_path)
        else:
            print("The clip has no audio track.")
        clip.close()
        print(
            f" ---> The extracted audio file is made with this path : [bold green]{extracted_audio_file_path}[/bold green]"
        )
    return extracted_audio_file_path


def audio_to_text(
    audio_file_path: str,
    language: str = "en",
) -> str:
    model = whisper.load_model("turbo")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        use_fp16 = True
    else:
        use_fp16 = False
    print(f"Using device: {device}")

    # Run transcription

    subscene_in_source_language_path = os.path.splitext(audio_file_path)[0] + ".txt"
    if os.path.exists(subscene_in_source_language_path):
        print(
            f" ---> The file [bold blue]{subscene_in_source_language_path}[/bold blue] is already exist"
        )
    else:
        result = model.transcribe(
            audio_file_path,
            fp16=use_fp16,
            language=language,
            verbose=True,
        )
        segments = result.get("segments")
        if not segments:
            text = result.get("text", "")
            return [("00:00.000", None, text.strip())]

        for seg in segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = seg.get("text", "").strip()
            with open(
                file=subscene_in_source_language_path, mode="a", encoding="utf8"
            ) as f:
                f.write(f"{start} ---> {end} ---> {text}\n")
        print(
            f" ---> The file [bold green]{subscene_in_source_language_path}[/bold green] is created"
        )
    return subscene_in_source_language_path


def ollama_models_info_list():
    models_list = ollama.list().models
    models_info = []
    # make a list of tupples that ready for Gradio choosing elements
    for model in models_list:
        models_info.append(
            (
                f"'{model.model.upper()}' with {model.size.human_readable()} size ",
                model.model,
            )
        )

    return models_info


def make_translated_subscene(input_file_path: str) -> str:
    subscene_in_target_language_path = (
        os.path.splitext(input_file_path)[0] + "_translated" + ".txt"
    )
    if os.path.exists(path=subscene_in_target_language_path):
        print(
            f" ---> The translated subscene file is already exist in : [bold blue]{subscene_in_target_language_path}[/bold blue]"
        )
    else:
        with (
            open(input_file_path, "r", encoding="utf-8") as infile,
            open(subscene_in_target_language_path, "a", encoding="utf-8") as outfile,
        ):

            print(f"Reading from '{input_file_path}'...")
            print(f"Writing to '{subscene_in_target_language_path}'...")

            # Iterate over each line in the input file
            for line in infile:
                # Strip any leading/trailing whitespace from the line
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                # Split the line into parts based on the " ---> " delimiter
                parts = line.split(" ---> ")
                print(parts)
                # Ensure the line has the expected format (3 parts)
                if len(parts) == 3:
                    start_time = parts[0]
                    end_time = parts[1]
                    text = parts[2]
                    modified_text = translate_text(input_text=text)
                    new_line = f"{start_time} ---> {end_time} ---> {modified_text}\n"
                    print(new_line)
                    outfile.write(new_line)
                else:
                    print(f"Warning: Skipping malformed line: {line}")
                    outfile.write(line + "\n")
            print(
                f" ---> The translated subscene file is made in : [bold green]{subscene_in_target_language_path}[/bold green]"
            )
    return subscene_in_target_language_path


def add_subtitles_to_video(
    subcene_file_path: str,
    input_video_path: str,
):
    output_video_path = (
        os.path.splitext(input_video_path)[0] + "_with_translated_subscene.mp4"
    )
    video = moviepy.VideoFileClip(input_video_path)

    subtitle_clips = []

    with open(file=subcene_file_path, mode="r", encoding="utf-8") as infile:

        print(f"Reading from '{subcene_file_path}'...")

        # Iterate over each line in the input file
        for line in infile:
            # Strip any leading/trailing whitespace from the line
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Split the line into parts based on the " ---> " delimiter
            parts = line.split(" ---> ")
            print(parts)
            # Ensure the line has the expected format (3 parts)
            if len(parts) == 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                text = parts[2]
                txt_clip = (
                    moviepy.TextClip(
                        text=text,
                        font_size=40,
                        color="white",
                        font="arial",
                        stroke_color="black",
                        stroke_width=1.5,
                    )
                    .with_position(("center", "bottom"))
                    .with_duration(end_time - start_time)
                    .with_start(start_time)
                )
            subtitle_clips.append(txt_clip)
        print(subtitle_clips)
        final_video = moviepy.CompositeVideoClip([video] + subtitle_clips)
        final_video.write_videofile(
            output_video_path, codec="libx264", audio_codec="aac"
        )
        print(
            f"ویدیو با موفقیت در مسیر [bold green]{output_video_path}[/bold green] ذخیره شد."
        )


def get_youtube_formats(url: str) -> list[str] | None:

    formatted_list = []

    try:
        # Create a YoutubeDL object and extract info without downloading
        with yt_dlp.YoutubeDL() as ydl:
            info_dict = ydl.extract_info(url, download=False)

            # The 'formats' key contains a list of dictionaries
            formats = info_dict.get("formats", [])

            # --- Optional: Print a header ---
            header = (
                f"{'ID':<10} {'EXT':<6} {'RESOLUTION':<15} {'FPS':<5} "
                f"{'VCODEC':<20} {'ACODEC':<15} {'SIZE (MB)':>12}"
            )
            print(header)
            print("-" * len(header))
            # ---------------------------------

            for f in formats:
                # Get file size, prefer actual over approximate
                filesize_bytes = f.get("filesize") or f.get("filesize_approx")
                filesize_mb = (
                    f"{filesize_bytes / (1024 * 1024):.2f}" if filesize_bytes else "N/A"
                )

                # Format the string with relevant information
                format_str = (
                    f"{f.get('format_id', 'N/A'):<10} "
                    f"{f.get('ext', 'N/A'):<6} "
                    f"{f.get('resolution', 'audio only'):<15} "
                    f"{str(f.get('fps', 'N/A')):<5} "
                    f"{f.get('vcodec', 'none'):<20} "
                    f"{f.get('acodec', 'none'):<15} "
                    f"{filesize_mb:>12}"
                )
                formatted_list.append(format_str)

    except yt_dlp.utils.DownloadError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return formatted_list


def embed_subtitle(video_path, subtitle_path):
    """
    Embeds an SRT subtitle file into a video file.

    The function creates a new video file with the subtitles "burned in"
    to the video frames. It aims to preserve the original video's quality.

    Args:
        video_path (str): The full path to the input video file.
        subtitle_path (str): The full path to the .srt subtitle file.

    Returns:
        str: The path to the newly created video file with subtitles,
             or None if an error occurred.
    """
    print(f"Starting to process video: {video_path}")
    print(f"With subtitles: {subtitle_path}")

    # Check if input files exist
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None
    if not os.path.exists(subtitle_path):
        print(f"Error: Subtitle file not found at {subtitle_path}")
        return None

    try:
        # Load the original video clip
        video_clip = moviepy.VideoFileClip(video_path)

        # Define the output path for the new video
        # It will be named like 'my_movie_subtitled.mp4'
        path_parts = os.path.splitext(video_path)
        output_path = f"{path_parts[0]}_subtitled.mp4"

        print(f"Output will be saved to: {output_path}")

        # Create a generator for the SubtitlesClip.
        # This function is more robust for different text encodings.
        # You can customize the font and fontsize here.
        # For Persian/Arabic subtitles, you might need a font like 'Tahoma' or 'Vazirmatn'.
        generator = lambda txt: moviepy.TextClip(
            txt,
            font="arial",  # A commonly available font
            font_size=24,
            color="white",
            stroke_color="black",
            stroke_width=0.5,
        )

        # Create the subtitles clip
        subtitles_clip = SubtitlesClip(subtitle_path, generator, encoding="utf-8")

        # Composite the video and subtitles
        # The subtitles will be placed at the bottom of the video
        final_clip = moviepy.CompositeVideoClip(
            [video_clip, subtitles_clip.with_position(("center", "bottom"))]
        )

        # Write the final video file to disk
        # We specify codecs and parameters to maintain quality.
        # 'libx264' is a high-quality video codec.
        # 'aac' is a standard audio codec.
        # The 'preset' affects encoding speed vs. file size. 'medium' is a good balance.
        # The original video's fps is used to ensure smoothness.
        print("Writing final video file... This may take a while.")
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            preset="medium",
            fps=video_clip.fps,
        )

        print("Successfully embedded subtitles!")
        return output_path

    except Exception as e:
        print(f"An error occurred during the process: {e}")
        return None


embed_subtitle(
    video_path="./Movies/How to Scrape Data From Any Website Using Deepseek.mp4",
    subtitle_path="Movies/How to Scrape Data From Any Website Using Deepseek.srt",
)
