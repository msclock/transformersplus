from typing import Mapping
import gradio as gr
import transformers
from transformers import AutoTokenizer


MODEL_URL = "wptoux/albert-chinese-large-qa"
MAX_LENGTH = 128  # depends on the model
tokenizer = AutoTokenizer.from_pretrained(MODEL_URL)


def is_triton_url(model_url: str):
    from urllib.parse import urlparse

    try:
        url = urlparse(model_url)
        return all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
    except ValueError:
        return False


def albert_masklm(
    question: gr.inputs.Textbox = None,
    context: gr.inputs.Textbox = None,
    model: gr.inputs.Dropdown = None,
):
    if is_triton_url(model):
        from transformersplus.utils.triton import TritonModel

        model_backend = TritonModel(model)
    else:
        from transformers import AutoModelForQuestionAnswering

        model_backend = AutoModelForQuestionAnswering.from_pretrained(model)
        model_backend.config.return_dict = True

    inputs = tokenizer(
        question,
        context,
        padding=transformers.utils.PaddingStrategy.MAX_LENGTH,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=transformers.TensorType.PYTORCH,
    )
    outputs = model_backend(**inputs)
    if isinstance(outputs, Mapping):
        start_logits, end_logits = outputs["start_logits"], outputs["end_logits"]
    else:
        start_logits, end_logits = outputs
    token_start_index = start_logits.argmax(dim=-1)
    token_end_index = end_logits.argmax(dim=-1)
    pred_ids = inputs["input_ids"][0][token_start_index : token_end_index + 1]
    prediction = tokenizer.decode(pred_ids)
    return prediction


app = gr.Interface(
    fn=albert_masklm,
    inputs=[
        gr.components.Textbox(label="Input question"),
        gr.components.Textbox(label="Input context"),
        gr.components.Dropdown(
            choices=[
                MODEL_URL,
                "http://localhost:8000/v2/models/albert_qa",
            ],
            value=MODEL_URL,
            label="Model",
        ),
    ],
    outputs=gr.components.Label(),
    title="albert qa",
    examples=[
        ["我住在哪里？", "我叫萨拉，我住在伦敦。", MODEL_URL],
    ],
)

app.launch(
    debug=True,
    enable_queue=True,
    share=False,
)
