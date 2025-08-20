import moviepy
import os
import yt_dlp
from rich import print
import whisper
import torch
import ollama
import time

# def audio_to_text(audio_file_path: str):
#     model = whisper.load_model("turbo")
#     audio = whisper.load_audio(audio_file_path)

#     audio = whisper.pad_or_trim(audio)

#     # Convert to log-Mel spectrogram
#     mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

#     # Detect the spoken language (optional)
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")

#     # Decode and transcribe
#     options = whisper.DecodingOptions()
#     result = whisper.decode(model, mel, options)

#     # Print the transcribed text
#     print(result.text)

os.system(command="cls" if os.name == "nt" else "clear")
print(moviepy.config.check())


def youtube_download(
    url: str,
    output_folder_path: str,
    quality: str = "bestvideo+bestaudio/best",
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

    print(final_file_path)
    return final_file_path


def audio_extractor(filename: str):
    clip = moviepy.VideoFileClip(filename=filename)
    if clip.audio is not None:
        clip.audio.write_audiofile(filename=os.path.splitext(filename)[0] + ".mp3")
    else:
        print("The clip has no audio track.")
    clip.close()
    print(filename)
    return filename


def format_timestamp(seconds: float) -> str:
    """Converts seconds to MM:SS.ms format."""
    if seconds is None:
        return "00:00.000"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:06.3f}"


def audio_to_text(
    audio_file_path: str,
    language: str = "en",
) -> list[tuple]:
    model = whisper.load_model("turbo")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        use_fp16 = True
    else:
        use_fp16 = False
    print(f"Using device: {device}")

    # Run transcription
    result = model.transcribe(
        audio_file_path, fp16=use_fp16, language=language, verbose=True
    )

    segments = result.get("segments")
    if not segments:
        text = result.get("text", "")
        return [("00:00.000", None, text.strip())]

    output = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = seg.get("text", "").strip()
        output.append((format_timestamp(start), format_timestamp(end), text))

    return output


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


def make_translated_subscene(
    input_list_of_tuples: list[tuple], output_file_name: str = "translated_subscene.txt"
) -> list[tuple]:
    list_length = len(input_list_of_tuples)
    translated_list = []
    for index, each_tuple in enumerate(input_list_of_tuples):
        translated_sentence = translate_text(input_text=each_tuple[2])
        translated_list.append((each_tuple[0], each_tuple[1], translated_sentence))
        print(
            f"sentenc [bold red]{index}[/bold red] from {list_length} sentences is translated"
        )
        with open(file=output_file_name, mode="a", encoding="utf8") as f:
            f.write(
                f"start:{each_tuple[0]} end:{each_tuple[1]} : {translated_sentence} \n"
            )
    print(
        f"translation is [bold green]Finished[/bold green] and the file {output_file_name} is created"
    )
    return translated_list


result = audio_to_text(
    audio_file_path="./videoclips/firstfile/Consumerism_is_the_Perfection_of_Slavery_Prof_Jiang_Xueqin.mp3"
)

final_result = make_translated_subscene(input_list_of_tuples=result)

for i in final_result:
    print(i)
