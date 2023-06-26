import typing
from urllib.parse import urlparse
import numpy as np
import torch
from tritonclient.utils import triton_to_np_dtype


class TritonModelClient:
    """A wrapper over a model served by the triton-server. It can be configured to communicate over GRPC or HTTP. It accept Torch Tensors as input and returns then as outputs"""

    def __init__(self, model_url) -> None:
        """
        Args:
            model url: Fully address of the tritonserver, such as::

                - grpc://localhost:8000/v2/path/to/model
                - http://localhost:8000/v2/path/to/model

        """
        super().__init__()
        parsed_url = urlparse(model_url)
        if parsed_url.scheme == "grpc":
            from tritonclient.grpc import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)  # Triton GRPC client
            model_repository = self.client.get_model_repository_index()
            self.model_name = model_repository.models[0].name
            self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)

            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(i["name"], [int(s) for s in i["shape"]], i["datatype"]) for i in self.metadata["inputs"]
                ]

        else:
            from tritonclient.http import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)  # Triton HTTP client
            model_repository = self.client.get_model_repository_index()
            self.model_name = model_repository[0]["name"]
            self.metadata = self.client.get_model_metadata(self.model_name)

            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(i["name"], [int(s) for s in i["shape"]], i["datatype"]) for i in self.metadata["inputs"]
                ]

        self._create_input_placeholders_fn = create_input_placeholders

    @property
    def runtime(self):
        """Returns the model runtime"""
        return self.metadata.get("backend", self.metadata.get("platform"))

    def __call__(self, *args, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]:
        """Invokes the model. Parameters can be provided via args or kwargs.
        args, if provided, are assumed to match the order of inputs of the model.
        kwargs are matched with the model input names.
        """
        inputs = self._create_inputs(*args, **kwargs)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)
        result = []
        for output in self.metadata["outputs"]:
            tensor = torch.as_tensor(response.as_numpy(output["name"]))
            result.append(tensor)
        return result[0] if len(result) == 1 else result

    def _create_inputs(self, *args, **kwargs):
        args_len, kwargs_len = len(args), len(kwargs)
        if not args_len and not kwargs_len:
            raise RuntimeError("No inputs provided.")
        if args_len and kwargs_len:
            raise RuntimeError("Cannot specify args and kwargs at the same time")

        def infer_input_type_from_data_type(input, val_shape):
            input_shape = input.shape()
            for idx, iv in enumerate(input_shape):
                if iv == -1:
                    input_shape[idx] = val_shape[idx]
            input.set_shape(input_shape)

        def ensure_data_type(input, value):
            np_t = triton_to_np_dtype(input.datatype())
            np_data: np = value.cpu().numpy()
            if self.runtime == "tensorrt_plan" and np_t != np_data.dtype:
                np_data = np_data.astype(np_t)
            return np_data

        placeholders = self._create_input_placeholders_fn()
        if args_len:
            if args_len != len(placeholders):
                raise RuntimeError(f"Expected {len(placeholders)} inputs, got {args_len}.")
            for input, value in zip(placeholders, args):
                infer_input_type_from_data_type(input, value.shape)
                input.set_data_from_numpy(ensure_data_type(input, value))
        else:
            for input in placeholders:
                value = kwargs[input.name()]
                infer_input_type_from_data_type(input, value.shape)
                input.set_data_from_numpy(ensure_data_type(input, value))
        return placeholders


class TritonModel(torch.nn.Module):
    base_model_prefix = "triton"

    def __init__(self, url) -> None:
        super().__init__()
        from transformersplus.utils.triton import TritonModelClient

        self.model = TritonModelClient(model_url=url)

    def forward(self, *input, **kwargs):
        return self.model(*input, **kwargs)
