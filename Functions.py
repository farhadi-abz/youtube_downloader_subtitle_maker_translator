import moviepy
import os
import yt_dlp
from rich import print
import whisper
import torch
import ollama
import time


os.system(command="cls" if os.name == "nt" else "clear")
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


# def format_timestamp(seconds: float) -> str:
#     """Converts seconds to MM:SS.ms format."""
#     if seconds is None:
#         return "00:00.000"
#     minutes = int(seconds // 60)
#     remaining_seconds = seconds % 60
#     return f"{minutes:02d}:{remaining_seconds:06.3f}"


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
    return translated_text


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


def add_subtitles_to_video(subtitles_list, input_video_path, output_video_path):
    """
    زیرنویس‌ها را از یک لیست تاپل (شامل زمان شروع، زمان پایان و متن) به یک ویدیو اضافه می‌کند.

    Parameters:
    - subtitles_list (list): لیستی از تاپل‌ها که هر تاپل شامل (زمان شروع، زمان پایان، متن) است.
    - input_video_path (str): مسیر فایل ویدیوی اصلی.
    - output_video_path (str): مسیر فایل ویدیوی خروجی با زیرنویس.
    """

    # بارگذاری کلیپ ویدیوی اصلی
    try:
        video = moviepy.VideoFileClip(input_video_path)
    except Exception as e:
        print(f"خطا در بارگذاری فایل ویدیو: {e}")
        return

    # یک لیست برای نگهداری کلیپ‌های متنی
    subtitle_clips = []

    # یک حلقه برای ساخت کلیپ متنی برای هر زیرنویس
    for start_time, end_time, text in subtitles_list:
        try:
            txt_clip = (
                moviepy.TextClip(
                    text,
                    fontsize=40,
                    color="white",
                    font="arial",
                    stroke_color="black",
                    stroke_width=1.5,
                )
                .set_position(("center", "bottom"))
                .set_duration(end_time - start_time)
                .set_start(start_time)
            )

            subtitle_clips.append(txt_clip)
        except Exception as e:
            print(f"خطا در ساخت کلیپ زیرنویس برای متن '{text}': {e}")
            continue

    # ترکیب ویدیو اصلی با کلیپ‌های زیرنویس
    final_video = moviepy.CompositeVideoClip([video] + subtitle_clips)

    # نوشتن ویدیو نهایی
    try:
        final_video.write_videofile(
            output_video_path, codec="libx264", audio_codec="aac"
        )
        print(f"ویدیو با موفقیت در مسیر {output_video_path} ذخیره شد.")
    except Exception as e:
        print(f"خطا در ذخیره ویدیو: {e}")


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


video_file_path = youtube_download(
    url="https://youtu.be/meEXag1XCFw?si=VLWys3Ae6FZ4K_NS"
)
audio_extracted_file_path = audio_extractor(file_path=video_file_path)
source_language_subscene = audio_to_text(
    audio_file_path=audio_extracted_file_path, language="en"
)

target_language_subscene = make_translated_subscene(
    input_file_path=source_language_subscene
)

# add_subtitles_to_video(
#     subtitles_list=target_language_subscene,
#     input_video_path=video_file_path,
#     output_video_path="./Movies_with_subtitle/",
# )

# test = audio_to_text(
#     audio_file_path="./Movies/جای آیفون‌های تاریخ مصرف‌گذشته اینارو بگیر.mp3",
#     language="fa",
# )
# print(test)
