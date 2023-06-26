## Albert Demo

### Prepare model

Generate model of the tritonserver with model_navigator script.

```bash
docker run -it --rm --gpus all -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:<yy.mm>-py3 bash -c '\
    pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator && \
    ./optimize.py --max-sequence-length=128 \
        --device=0 \
        --model-repository=.model_repository'
```

### Launch server and test locally

Launch tritonserver with the local generated model repository in the folder `.model_repository`.

```bash
docker run -it --gpus=all \
    --shm-size=256m \
    --rm \
    -p8000:8000 -p8001:8001 -p8002:8002 \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/tritonserver:<yy.mm>-py3 \
    bash -c 'tritonserver --model-repository=.model_repository --model-control-mode=explicit --load-model=albert_masklm'
```

Test with full functional app-ui based on Gradio.
```bash
python app.py
```

![Alt text](image.png)

Then select triton model and input a mask chinese text.