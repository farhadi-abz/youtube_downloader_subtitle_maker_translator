import Functions
import gradio as gr
import os


def get_dynamic_choices_for_ollama_models():
    ollama_models = Functions.ollama_models_info_list()
    return gr.Radio(label="choose your llm for translation:", choices=ollama_models)


def get_dynamic_choices_for_whisper_languages():
    whisper_languages = Functions.get_all_whisper_languages_supports()
    return gr.Dropdown(
        label="choose your subtitle language: ", choices=whisper_languages
    )


def main():
    os.system(command="cls" if os.name == "nt" else "clear")
    with gr.Blocks() as gui:
        gr.Markdown("# Gradio subtitle translation Tool")
        choose = gr.Radio(
            label="Choose function you want: ",
            choices=[
                "youtube_downloader",
                "extract_subtitle_from_video",
                "translate_subtitle",
            ],
            value="youtube_downloader",
        )

        @gr.render(inputs=choose)
        def render_app(method):
            if method == "youtube_downloader":
                with gr.Column():
                    input_youtube_video_address = gr.Textbox(label="Youtube URL:")
                    input_video_quality = gr.Radio(
                        label="Video Quality:",
                        choices=["audio", "480p", "720p", "1080p"],
                    )
                    btn_youtube_downloader = gr.Button("Start Download")
                    output_video_player = gr.Video()
                    btn_youtube_downloader.click(
                        fn=Functions.youtube_downloader,
                        inputs=[input_youtube_video_address, input_video_quality],
                        outputs=output_video_player,
                    )
            elif method == "extract_subtitle_from_video":
                with gr.Column():
                    input_video_address = gr.File(
                        label="Uploade the video file you want to subtitle:"
                    )
                    input_subtitle_language = gr.Dropdown()
                    btn_extract_subtitle = gr.Button("Start Extract Subtitle")
                    output_subtitle_file = gr.File(label="This is the subtitle file")
                    btn_extract_subtitle.click(
                        fn=Functions.generate_srt_from_video_by_whisper,
                        inputs=[input_video_address, input_subtitle_language],
                        outputs=output_subtitle_file,
                    )
                gui.load(
                    fn=get_dynamic_choices_for_whisper_languages,
                    inputs=None,
                    outputs=input_subtitle_language,
                )
            elif method == "translate_subtitle":
                with gr.Column():
                    input_ollama_model_for_translation = gr.Radio(interactive=True)
                    input_subtitle_file = gr.File(
                        label="Upload the subtitle you want to translate:"
                    )
                    input_source_language = gr.Radio(
                        choices=[
                            "English",
                            "Persian(Farsi)",
                            "Japanes",
                            "Arabic",
                            "Chineas",
                            "German",
                            "Russian",
                            "Spanish",
                            "French",
                        ],
                        label="choose your source language :",
                    )
                    input_taget_language = gr.Radio(
                        choices=[
                            "English",
                            "Persian(Farsi)",
                            "Japanes",
                            "Arabic",
                            "Chineas",
                            "German",
                            "Russian",
                            "Spanish",
                            "French",
                        ],
                        label="choose your target language :",
                    )

                    btn_translate_subtitle = gr.Button("Start Translate Subtitle")
                    output_translated_subtitle_file = gr.File(
                        label="Download your translated subtitle file: "
                    )
                    btn_translate_subtitle.click(
                        fn=Functions.translate_srt_file,
                        inputs=[
                            input_ollama_model_for_translation,
                            input_subtitle_file,
                            input_source_language,
                            input_taget_language,
                        ],
                        outputs=output_translated_subtitle_file,
                    )
                gui.load(
                    fn=get_dynamic_choices_for_ollama_models,
                    inputs=None,
                    outputs=input_ollama_model_for_translation,
                )

    gui.launch(allowed_paths=["./Movies/"])


if __name__ == "__main__":
    main()
