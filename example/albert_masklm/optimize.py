#!/usr/bin/env python3

import os

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import itertools
import pathlib
import string

import numpy as np
from transformers import AutoTokenizer, TensorType
from transformers.utils import PaddingStrategy
from transformers import AutoModelForMaskedLM
from transformers.onnx.features import FeaturesManager
import model_navigator as nav


def get_model(model_name: str):
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.config.return_dict = False  # return one value from the inference method
    return model


def get_dataloader(
    model_name: str,
    max_sequence_length: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if max_sequence_length == -1:
        max_sequence_length = getattr(tokenizer, "model_max_length", 512)

    if max_sequence_length > 512:
        max_sequence_length = 512

    num_samples = 10

    return [
        tokenizer(
            "".join(
                np.random.choice(list(string.ascii_letters + string.digits), size=32)
            ),
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors=TensorType.PYTORCH,
        )
        for _ in range(num_samples)
    ]


def get_verify_function():
    def verify_func(ys_runner, ys_expected):
        """Verify that at least 99% max probability tokens match on any given batch."""
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                np.mean(a.argmax(axis=2) == b.argmax(axis=2)) > 0.99
                for a, b in zip(y_runner.values(), y_expected.values())
            ):
                return False
        return True

    return verify_func


def get_configuration(
    model_name: str,
    batch_size: int,
    max_sequence_length: int,
):
    model = FeaturesManager.get_model_from_feature(
        model=model_name,
        feature="masked-lm",
    )
    _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
        model=model,
        feature="masked-lm",
    )
    onnx_config = model_onnx_config(model.config)
    input_names = tuple(onnx_config.inputs.keys())
    output_names = tuple(onnx_config.outputs.keys())
    dynamic_axes = {
        name: axes
        for name, axes in itertools.chain(
            onnx_config.inputs.items(),
            onnx_config.outputs.items(),
        )
    }
    opset = onnx_config.default_onnx_opset

    tensorrt_profile = nav.TensorRTProfile()
    for k in input_names:
        tensorrt_profile.add(
            k,
            (1, max_sequence_length),
            (1, max_sequence_length),
            (batch_size, max_sequence_length),
        )

    optimization_profile = nav.OptimizationProfile(
        max_batch_size=batch_size,
        batch_sizes=[
            1,
            batch_size,
        ],
        stability_percentage=15,
        max_trials=5,
        throughput_cutoff_threshold=0.1,
    )

    configuration = {
        "input_names": input_names,
        "output_names": output_names,
        "sample_count": 10,
        "optimization_profile": optimization_profile,
        "custom_configs": [
            nav.TorchConfig(
                jit_type=nav.JitType.TRACE,
                strict=False,
            ),
            nav.OnnxConfig(
                opset=opset,
                dynamic_axes=dynamic_axes,
            ),
            nav.TensorRTConfig(
                precision=(nav.TensorRTPrecision.FP32),
                max_workspace_size=2 * 1024 * 1024 * 1024,
                trt_profile=tensorrt_profile,
            ),
        ],
    }
    return configuration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=str,
        default=".navigator_workspace",
        help="navigator cache workspace",
    )
    parser.add_argument(
        "--input-model",
        type=str,
        default="uer/albert-base-chinese-cluecorpussmall",
        help="input model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="albert_masklm",
        help="sub dir model name in model store model_repository folder",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="batch size on model",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=-1,
        help="max input text sequence length on model, up to 512",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device = None or 'cpu' or 0 or '0' or '0,1,2,3'",
    )
    parser.add_argument(
        "--model-repository",
        type=str,
        default=f".model_repository",
        help="model repository folder serving on tirton",
    )
    return parser.parse_args()


def main(FLAGS):
    model = get_model(FLAGS.input_model)
    dataloader = get_dataloader(
        model_name=FLAGS.input_model,
        max_sequence_length=FLAGS.max_sequence_length,
    )
    verify_func = get_verify_function()
    configuration = get_configuration(
        model_name=FLAGS.input_model,
        batch_size=FLAGS.batch_size,
        max_sequence_length=FLAGS.max_sequence_length,
    )

    package = nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        verify_func=verify_func,
        target_device=nav.DeviceKind.CPU
        if str(FLAGS.device) == "cpu"
        else nav.DeviceKind.CUDA,
        debug=True,
        verbose=True,
        workspace=pathlib.Path(FLAGS.workspace) / FLAGS.model_name,
        **configuration,
    )

    import shutil

    shutil.rmtree(
        pathlib.Path(FLAGS.model_repository) / FLAGS.model_name,
        ignore_errors=True,
    )
    nav.triton.model_repository.add_model_from_package(
        model_repository_path=pathlib.Path(FLAGS.model_repository),
        model_name=FLAGS.model_name,
        package=package,
        strategy=nav.MaxThroughputStrategy(),
    )


if __name__ == "__main__":
    main(parse_args())
