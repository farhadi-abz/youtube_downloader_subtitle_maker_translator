import gradio as gr
from gradio.themes.utils import colors, fonts, sizes


website_theme = gr.themes.Soft(
    primary_hue=colors.orange,
    secondary_hue=colors.sky,
    neutral_hue=colors.gray,
    font=(fonts.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
    radius_size=sizes.radius_lg,  # MOVED: radius_size is a constructor argument
).set(
    # You can optionally override specific details of the theme
    # For example, let's make the body background a slightly different shade
    body_background_fill="#F5F5F5",
    body_background_fill_dark="#111111",
    # Customize button styles
    button_primary_background_fill=colors.orange.c500,
    button_primary_background_fill_hover=colors.orange.c400,
    button_primary_text_color="white",
    button_secondary_background_fill=colors.sky.c500,
    button_secondary_background_fill_hover=colors.sky.c400,
    button_secondary_text_color="white",
)


def analyze_text(text, sentiment_analysis, entity_recognition):
    """
    A dummy function to simulate text analysis based on user choices.
    """
    results = f"--- Analysis for: '{text}' ---\n"
    if sentiment_analysis:
        results += "Sentiment: Positive (92% confidence)\n"
    if entity_recognition:
        results += "Entities Recognized: [Gradio (Organization), Python (Technology)]\n"
    if not sentiment_analysis and not entity_recognition:
        return "Please select at least one analysis type."
    return results
