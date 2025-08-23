import moviepy
import os
import yt_dlp
from rich import print
import whisper
import torch
import ollama
import time
import re


def get_all_whisper_languages_supports():
    supported_languages = sorted(list(whisper.tokenizer.LANGUAGES.values()))
    return supported_languages


def ollama_models_info_list():
    models_list = ollama.list().models
    models_info = []
    # make a list of tupples that ready for Gradio choosing elements
    for model in models_list:
        models_info.append(
            (
                f"{model.model.upper()} ->({model.size.human_readable()}) ",
                model.model,
            )
        )

    return models_info


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


def youtube_downloader(url, quality_setting):
    """
    Downloads a video from the URL with specified quality, saves it to ./Movies,
    checks for existing files, and returns the final file path.
    """
    movies_dir = "Movies"
    musics_dir = "Musics"

    try:
        # Get video metadata (like the title) without downloading
        info_opts = {"quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "untitled_video")
            sanitized_title = sanitize_filename(title)

        # --- START OF CORRECTION ---

        # 1. Define the base path WITHOUT the extension
        if quality_setting == "audio":
            os.makedirs(musics_dir, exist_ok=True)
            base_filepath = os.path.join(musics_dir, sanitized_title)
        else:
            os.makedirs(movies_dir, exist_ok=True)
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


def generate_srt_from_video_by_whisper(
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
    srt_filename = srt_filename.replace("\\", "/")

    if os.path.exists(path=srt_filename):
        print(f"[bold blue]The file already exist[/bold blue]")
        return srt_filename
    else:
        print(f"starting transcription for '{video_path}'...")
        # whisper handles audio extraction automatically from video files.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            use_fp16 = True
        else:
            use_fp16 = False
        result = model.transcribe(
            video_path,
            verbose=True,
            fp16=use_fp16,
            language=language,
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


def translate_srt_file(
    ollama_model: str,
    file_path: str,
    source_language: str = "English",
    target_language: str = "Persian(Farsi)",
):
    file_path = file_path.replace("\\", "/")
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
    blocks_total_number = len(subtitle_blocks)
    translated_blocks = []

    for index, block in enumerate(subtitle_blocks):
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
        translated_text = translate_text(
            input_text=original_text,
            model=ollama_model,
            source_lang=source_language,
            target_lang=target_language,
        )

        # Reconstruct the block with the translated text
        new_block = f"{sequence_number}\n{timestamp}\n{translated_text}"
        translated_blocks.append(new_block)
        print(
            f"translate block number: [bold green]{index}[/bold green] from [bold yellow]{blocks_total_number}[/bold yellow] blocks"
        )
    # Join the translated blocks back together
    new_content = "\n\n".join(translated_blocks)

    try:
        # Write the new content back to the original file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Successfully translated and updated the file: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error writing the translated content to the file: {e}")
