import gradio as gr
import torch
from torch.nn.functional import softmax
import transformers
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("uer/albert-base-chinese-cluecorpussmall")


def is_triton_url(model_url: str):
    from urllib.parse import urlparse

    try:
        url = urlparse(model_url)
        return all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
    except ValueError:
        return False


def albert_masklm(
    text: gr.inputs.Textbox = None,
    model: gr.inputs.Dropdown = None,
):
    if is_triton_url(model):
        from transformersplus.utils.triton import TritonModel

        model_backend = TritonModel(model)
    else:
        from transformers.models.albert.modeling_albert import AlbertForMaskedLM

        model_backend = AlbertForMaskedLM.from_pretrained(model)

    inputs = tokenizer(
        [text],
        padding=transformers.utils.PaddingStrategy.MAX_LENGTH,
        truncation=True,
        max_length=128,
        return_tensors=transformers.TensorType.PYTORCH,
    )
    input_ids = inputs["input_ids"][0]
    masked_index = torch.nonzero(
        input_ids == tokenizer.mask_token_id,
        as_tuple=False,
    ).squeeze(-1)
    outputs = model_backend(**inputs)
    if isinstance(outputs, transformers.modeling_outputs.MaskedLMOutput):
        outputs = outputs["logits"]
    logits = outputs[0, masked_index, :]
    probs = logits.softmax(dim=-1)
    values, predictions = probs.topk(6)
    results = [
        [(tokenizer.decode([p]), v) for v, p in zip(_values, _predictions)]
        for _values, _predictions in zip(values.tolist(), predictions.tolist())
    ]
    return {k: v for k, v in results[0]}


app = gr.Interface(
    fn=albert_masklm,
    inputs=[
        gr.components.Textbox(label="Mask input"),
        gr.components.Dropdown(
            choices=[
                "uer/albert-base-chinese-cluecorpussmall",
                "http://localhost:8000/v2/models/albert_masklm",
            ],
            value="uer/albert-base-chinese-cluecorpussmall",
            label="Model",
        ),
    ],
    outputs=gr.components.Label(),
    title="albert mask lm",
    examples=[
        ["今天[MASK]情很好", "uer/albert-base-chinese-cluecorpussmall"],
        ["马上下[MASK]了", "uer/albert-base-chinese-cluecorpussmall"],
    ],
    cache_examples=True,
)

app.launch(
    debug=True,
    enable_queue=True,
    share=False,
)
