#!/usr/bin/env python3

import math
import os

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import itertools
import pathlib

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TensorType
from transformers.models.albert.configuration_albert import AlbertOnnxConfig
from transformers.models.albert.modeling_albert import AlbertForMaskedLM

import model_navigator as nav


def get_model(model_name: str):
    model = AlbertForMaskedLM.from_pretrained(model_name)
    model.config.return_dict = True
    return model


def get_dataloader(
    model_name: str,
    dataset_name: str,
    max_batch_size: int,
    num_samples: int,
    max_sequence_length: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if max_sequence_length == -1:
        max_sequence_length = getattr(tokenizer, "model_max_length", 512)

    model_config = AutoConfig.from_pretrained(model_name)
    onnx_config = AlbertOnnxConfig(model_config)
    input_names = tuple(onnx_config.inputs.keys())
    dataset = load_dataset(dataset_name)["train"]

    def preprocess_function(examples):
        return tokenizer(
            examples["content"], truncation=True, max_length=max_sequence_length
        )

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(
        [c for c in tokenized_dataset.column_names if c not in input_names]
    )
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=max_batch_size,
        collate_fn=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_sequence_length,
            return_tensors=TensorType.PYTORCH,
        ),
    )

    return [sample for sample, _ in zip(dataloader, range(num_samples))]


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


def get_profiler_config(FLAGS):
    return nav.torch.ProfilerConfig(
        run_profiling=True,
        batch_sizes=[
            1,
            math.ceil(FLAGS.batch_size / 2),
            FLAGS.batch_size,
        ],
        measurement_mode=nav.MeasurementMode.TIME_WINDOWS,
        measurement_interval=2500,  # ms
        measurement_request_count=10,
        stability_percentage=15,
        max_trials=5,
        throughput_cutoff_threshold=0.1,
    )


def get_configuration(
    model_name: str,
    batch_size: int,
    max_sequence_length: int,
):
    model_config = AutoConfig.from_pretrained(model_name)
    onnx_config = AlbertOnnxConfig(model_config)
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
            (math.ceil(batch_size / 2), max_sequence_length),
            (batch_size, max_sequence_length),
        )

    configuration = {
        "input_names": input_names,
        "output_names": output_names,
        "sample_count": 10,
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
        help="max input text sequence length on model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device = None or 'cpu' or 0 or '0' or '0,1,2,3'",
    )
    parser.add_argument(
        "--min-top1-accuracy",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--model-repository",
        type=str,
        default=f".model_repository",
        help="model repository folder serving on tirton",
    )
    return parser.parse_args()


def main(FLAGS):
    dataset_name = "madao33/new-title-chinese"
    num_samples = 10

    model = get_model(FLAGS.input_model)
    dataloader = get_dataloader(
        model_name=FLAGS.input_model,
        dataset_name=dataset_name,
        max_batch_size=FLAGS.batch_size,
        num_samples=num_samples,
        max_sequence_length=FLAGS.max_sequence_length,
    )
    verify_func = get_verify_function()
    profiler_config = get_profiler_config(FLAGS)
    configuration = get_configuration(
        model_name=FLAGS.input_model,
        batch_size=FLAGS.batch_size,
        max_sequence_length=FLAGS.max_sequence_length,
    )

    package = nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        # verify_func=verify_func,
        target_device=nav.DeviceKind.CPU
        if str(FLAGS.device) == "cpu"
        else nav.DeviceKind.CUDA,
        debug=True,
        verbose=True,
        workspace=pathlib.Path(FLAGS.workspace) / FLAGS.model_name,
        profiler_config=profiler_config,
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
