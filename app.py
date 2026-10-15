import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokenizer import detokenize

REPO = os.environ.get("MVGPT_REPO", "nenuadrian/mvGPT")
# The labels the app shows, and the tags on the Hub they load.
MODELS = {"Fine-tuned (episode 3)": "ep003-sft", "Pretrained (episode 2)": "ep002-pretrain"}
FEATURES = ["Dialogue", "BadEnding", "MoralValue", "Twist", "Foreshadowing", "Conflict"]
MAX_TOKENS = 200

loaded = {}


def load(revision):
    # Each version is downloaded the first time it's asked for, and kept in memory after that.
    if revision not in loaded:
        tokenizer = AutoTokenizer.from_pretrained(REPO, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(REPO, revision=revision).eval()
        loaded[revision] = tokenizer, model
    return loaded[revision]


def instruction(features, words, summary):
    # The fields in the order sft.py gives them, ending where the story starts.
    lines = [f"Features: {', '.join(features)}"] if features else []
    if words.strip():
        lines.append(f"Words: {words.strip()}")
    if summary.strip():
        lines.append(f"Summary: {summary.strip()}")
    return "\n".join(lines + ["Story:"])


@torch.no_grad()
def write(version, features, words, summary, temperature, top_k):
    tokenizer, model = load(MODELS[version])
    x = tokenizer(instruction(features, words, summary), return_tensors="pt").input_ids
    past, story = None, []
    for _ in range(min(MAX_TOKENS, model.config.n_positions - x.size(1))):
        # transformers' GPT-2 has a KV cache like ours: the cache goes in, and comes back with x's.
        out = model(x, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1] / temperature
        logits[logits < torch.topk(logits, int(top_k)).values[-1]] = -float("inf")
        x = torch.multinomial(torch.softmax(logits, dim=-1), 1)[None]
        if x.item() == tokenizer.eos_token_id:
            break
        story.append(x.item())
        # Yielding the story so far makes Gradio show it a token at a time.
        yield detokenize(tokenizer.convert_ids_to_tokens(story))


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo, neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Space Grotesk"), font_mono=gr.themes.GoogleFont("JetBrains Mono"),
)
# Dark, whatever the visitor's system prefers: each colour takes its dark-mode value.
THEME.set(**{name.removesuffix("_dark"): getattr(THEME, name) for name in vars(THEME) if name.endswith("_dark")})
# The series' near-black slate, and its periwinkle for the button, in both modes.
COLOURS = dict(body_background_fill="#101215", block_background_fill="#191d22", input_background_fill="#101215",
               button_primary_background_fill="#9fb4ff", button_primary_background_fill_hover="#a8e6ed",
               button_primary_text_color="#101215")
THEME.set(**COLOURS, **{f"{name}_dark": value for name, value in COLOURS.items()})

with gr.Blocks(title="mvGPT") as demo:
    gr.Markdown("# mvGPT\nA 15-million-parameter GPT from the series "
                "[Minimum Viable GPT](https://mvgpt.nenuadrian.com), writing children's stories.")
    with gr.Row():
        with gr.Column():
            version = gr.Radio(list(MODELS), value=next(iter(MODELS)), label="Model")
            features = gr.CheckboxGroup(FEATURES, label="Features")
            words = gr.Textbox(label="Words", placeholder="dog, ball, happy")
            summary = gr.Textbox(label="Summary", placeholder="A dog makes a new friend at the park.")
            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_k = gr.Slider(1, 100, value=50, step=1, label="Top k")
            button = gr.Button("Write a story", variant="primary")
        story = gr.Textbox(label="Story", lines=16)
    button.click(write, [version, features, words, summary, temperature, top_k], story)

if __name__ == "__main__":
    # In Docker, the app has to listen on all interfaces for a published port to reach it.
    demo.launch(server_name=os.environ.get("HOST", "127.0.0.1"), server_port=7860, theme=THEME)
