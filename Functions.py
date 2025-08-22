import moviepy
import os
import yt_dlp
from rich import print
import whisper
import torch
import ollama
import time
import re

os.system(command="cls" if os.name == "nt" else "clear")
print(moviepy.config.check())


def sanitize_filename(filename):
    """
    Removes characters that are invalid in Windows and Linux filenames.
    """
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Replace whitespace sequences with a single space
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def get_ydl_options(quality="720p"):
    """
    Returns a dictionary of yt-dlp options based on the desired quality.
    Note: 'outtmpl' is now set in the main download function.
    """
    options = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        # This postprocessor ensures the final merged file is in an mp4 container.
        "postprocessors": [
            {
                "key": "FFmpegVideoRemuxer",
                "preferedformat": "mp4",
            }
        ],
    }

    if quality == "audio":
        # Options for best audio only. We prefer m4a which is an mp4 container.
        options.update(
            {
                "format": "bestaudio[ext=m4a]/bestaudio",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "aac",  # AAC is a common codec for mp4 audio
                    }
                ],
            }
        )
    elif quality == "480p":
        options["format"] = "bestvideo[height<=480]+bestaudio/best[height<=480]"
    elif quality == "720p":
        options["format"] = "bestvideo[height<=720]+bestaudio/best[height<=720]"
    elif quality == "1080p":
        options["format"] = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
    else:
        print(f"Warning: Invalid quality '{quality}'. Defaulting to 720p.")
        # Default is already set

    return options


def download_video(url, quality_setting):
    """
    Downloads a video from the URL with specified quality, saves it to ./Movies,
    checks for existing files, and returns the final file path.
    """
    movies_dir = "Movies"
    os.makedirs(movies_dir, exist_ok=True)

    try:
        # Get video metadata (like the title) without downloading
        info_opts = {"quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "untitled_video")
            sanitized_title = sanitize_filename(title)

        # --- START OF CORRECTION ---

        # 1. Define the base path WITHOUT the extension
        base_filepath = os.path.join(movies_dir, sanitized_title)

        # 2. Define the final path WITH extension for checking and returning
        final_filepath = base_filepath + ".mp4"

        if os.path.exists(final_filepath):
            print("The file already exist ")
            return final_filepath

        print(f"Starting download for: '{title}'")
        ydl_opts = get_ydl_options(quality_setting)

        # 3. Pass the base path (without extension) to yt-dlp's outtmpl
        ydl_opts["outtmpl"] = base_filepath

        # --- END OF CORRECTION ---

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        print(f"Download completed successfully! Saved to: {final_filepath}")
        return final_filepath

    except Exception as e:
        # Since you are in Iran, network errors can be common.
        # This will catch errors if the video is unavailable or the connection fails.
        if "HTTP Error 403" in str(e):
            print(
                f"An error occurred: Access to the video is forbidden (HTTP 403). It might be region-locked."
            )
        else:
            print(f"An error occurred: {e}")
        return None


def format_timestamp(seconds: float) -> str:
    """converts seconds to the srt timestamp format hh:mm:ss,ms."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def generate_srt_from_whisper(
    video_path: str,
    language: str = "en",
):
    """
    generates an srt file from a video using whisper.

    args:
        video_path (str): the path to the input video file.
        model_name (str): the name of the whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
    """
    if language == "en":
        model_name = "small.en"
    else:
        model_name: str = "turbo"
    if not os.path.exists(video_path):
        print(f"error: video file not found at '{video_path}'")
        return

    print(f"loading whisper model: '{model_name}'...")
    # on systems without a powerful gpu, you can force cpu usage
    # model = whisper.load_model(model_name, device="cpu")
    model = whisper.load_model(model_name)
    print("model loaded successfully.")

    # define the output srt filename
    srt_filename = os.path.splitext(video_path)[0] + ".srt"
    if os.path.exists(path=srt_filename):
        print(f"[bold blue]The file already exist[/bold blue]")
    else:
        print(f"starting transcription for '{video_path}'...")
        # whisper handles audio extraction automatically from video files.
        result = model.transcribe(
            video_path, verbose=True
        )  # verbose=true prints progress
        print("transcription completed.")

        print(f"saving subtitles to '{srt_filename}'...")
        with open(srt_filename, "w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(result["segments"]):
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"].strip()

                srt_file.write(f"{i + 1}\n")
                srt_file.write(f"{start_time} --> {end_time}\n")
                srt_file.write(f"{text}\n\n")

        print("srt file created successfully!")
    return srt_filename


def translate_text(
    input_text: str,
    model: str = "gemma3:12b",
    temprature: float = 0.0,
    source_lang: str = "English",
    target_lang: str = "Persian(Farsi)",
):
    SYSTEM_PROMPT: str = (
        f"""You are a professional translator model that translates text from {source_lang} to {target_lang} .
        Your translations must adhere to the following guidelines:
        Do not ask any question or include any comment just translate the pure text and return it without any question . 
        1. Language Exclusivity: Translate the provided text exclusively into {target_lang}.
        2. Do not include any {source_lang} words .
        2. Literary Quality: Ensure that the translation is literary, fluent, eloquent, and smooth. The text should read naturally and with high-level stylistic quality.
        3. Accurate Word Choice: Carefully select words that precisely capture the meaning and context of the original text. Each word must be both semantically accurate and contextually appropriate.
        4. Adherence to {target_lang} Grammar: Follow all principles of {target_lang} grammar, punctuation, and standard orthography without exception.
        5. Correct Use of Half-Spaces: Pay meticulous attention to the correct usage of half-spaces in {target_lang}.
        6. Your output should be a refined, accurate, and stylistically polished {target_lang} translation of the provided {source_lang} text.
        7. No {source_lang} Words: Do not use any {source_lang} words or phrases in your translation.
        """
    )
    start_time = time.time()  # to avoid rate limit
    response = ollama.chat(
        think=False,
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": input_text},
        ],
        options={"temprature": temprature},
        stream=False,
    )
    translated_text = response["message"]["content"]
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"translate in {elapsed_time} second time")
    return translated_text


def translate_srt_file(file_path: str):
    """
    Reads an SRT file, translates the text content of each subtitle,
    and overwrites the file with the translated version.

    Args:
        file_path (str): The full path to the .srt file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Starting translation for SRT file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Split the file into subtitle blocks. Blocks are separated by double newlines.
    subtitle_blocks = content.strip().split("\n\n")

    translated_blocks = []

    for block in subtitle_blocks:
        if not block.strip():
            continue

        lines = block.split("\n")

        # A valid block has at least 3 lines: number, timestamp, text
        if len(lines) < 3:
            # It might be an invalid block, so we keep it as is
            translated_blocks.append(block)
            continue

        # The first two lines are the sequence number and the timestamp
        sequence_number = lines[0]
        timestamp = lines[1]

        # The rest of the lines are the subtitle text
        original_text = "\n".join(lines[2:])

        # Translate the text using our placeholder function
        translated_text = translate_text(input_text=original_text, model="gemma3:4b")

        # Reconstruct the block with the translated text
        new_block = f"{sequence_number}\n{timestamp}\n{translated_text}"
        translated_blocks.append(new_block)

    # Join the translated blocks back together
    new_content = "\n\n".join(translated_blocks)

    try:
        # Write the new content back to the original file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Successfully translated and updated the file: {file_path}")
    except Exception as e:
        print(f"Error writing the translated content to the file: {e}")


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


movie_path = download_video(
    url="https://youtu.be/13Fe36L9sYQ?si=fCs0VebZnPYjrjX1", quality_setting="1080p"
)

srt_file_path = generate_srt_from_whisper(video_path=movie_path)

translate_srt_file(file_path=srt_file_path)
