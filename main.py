# ---
# deploy: true
# ---
# # Serverless TensorRT-LLM (LLaMA 3 8B)
#
# In this example, we demonstrate how to use the TensorRT-LLM framework to serve Meta's LLaMA 3 8B model
# at a total throughput of roughly 4,500 output tokens per second on a single NVIDIA A100 40GB GPU.
# At [Modal's on-demand rate](https://modal.com/pricing) of ~$4/hr, that's under $0.20 per million tokens --
# on auto-scaling infrastructure and served via a customizable API.
#
# Additional optimizations like speculative sampling and FP8 quantization can further improve throughput.
# For more on the throughput levels that are possible with TensorRT-LLM for different combinations
# of model, hardware, and workload, see the
# [official benchmarks](https://github.com/NVIDIA/TensorRT-LLM/blob/71d8d4d3dc655671f32535d6d2b60cab87f36e87/docs/source/performance.md).
#
# ## Overview
#
# This guide is intended to document two things:
# the general process for building TensorRT-LLM on Modal
# and a specific configuration for serving the LLaMA 3 8B model.
#
# ### Build process
#
# Any given TensorRT-LLM service requires a multi-stage build process,
# starting from model weights and ending with a compiled engine.
# Because that process touches many sharp-edged high-performance components
# across the stack, it can easily go wrong in subtle and hard-to-debug ways
# that are idiosyncratic to specific systems.
# And debugging GPU workloads is expensive!
#
# This example builds an entire service from scratch, from downloading weight tensors
# to responding to requests, and so serves as living, interactive documentation of a TensorRT-LLM
# build process that works on Modal.
#
# ### Engine configuration
#
# TensorRT-LLM is the Lamborghini of inference engines: it achieves seriously
# impressive performance, but only if you tune it carefully.
# We carefully document the choices we made here and point to additional resources
# so you know where and how you might adjust the parameters for your use case.
#
# ## Installing TensorRT-LLM
#
# To run TensorRT-LLM, we must first install it. Easier said than done!
#
# In Modal, we define [container images](https://modal.com/docs/guide/custom-container) that run our serverless workloads.
# All Modal containers have access to GPU drivers via the underlying host environment,
# but we still need to install the software stack on top of the drivers, from the CUDA runtime up.
#
# We start from the official `nvidia/cuda:12.1.1-devel-ubuntu22.04` image,
# which includes the CUDA runtime & development libraries
# and the environment configuration necessary to run them.

from typing import Optional

import modal
import pydantic  # for typing, used later
import os

tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
)

# Add some system dependencies of TensorRT-LLM,
# including OpenMPI for distributed communication, some core software like `git`,
# and the `tensorrt_llm` package itself.

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "tensorrt_llm==0.13.0.dev2024081300",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

MODEL_DIR = "/root/model/model_input"
MODEL_ID = "mistralai/Mamba-Codestral-7B-v0.1"
MODEL_REVISION = "d4521ac4b7658f796233ce47e8e695933f3cd48a"  # pin model revisions to prevent unexpected changes!
HF_TOKEN = os.environ["HF_TOKEN"]


def download_model():
    import os

    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        revision=MODEL_REVISION,
        token=HF_TOKEN,
    )
    move_cache()


MINUTES = 60  # seconds
tensorrt_image = (  # update the image by downloading the model we're using
    tensorrt_image.pip_install(  # add utilities for downloading the model
        "hf-transfer==0.1.8",
        "huggingface_hub==0.24.6",
        "requests~=2.31.0",
        "mistral-inference==1.3.1",
    )
    .env(  # hf-transfer: faster downloads, but fewer comforts
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    )
    .run_function(  # download the model
        download_model,
        timeout=20 * MINUTES,
    )
)

# todo: add hash
# this is pegged at bca9a33
# it's always nice to update stuff, but
# if anything breaks, at least bca9a33
# worked at some point
CHECKPOINT_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/mamba/convert_checkpoint.py"

GPU_CONFIG = modal.gpu.A100(count=1)

DTYPE = "float16"

# We put that all together with another invocation of `.run_commands`.

CKPT_DIR = "/root/model/model_ckpt"
tensorrt_image = tensorrt_image.run_commands(  # update the image by converting the model to TensorRT format  # takes ~5 minutes
    [
        f"wget {CHECKPOINT_SCRIPT_URL} -O /root/convert_checkpoint.py",
        f"python /root/convert_checkpoint.py --model_dir={MODEL_DIR} --output_dir={CKPT_DIR}"
        + f" --dtype={DTYPE}",
    ],
    gpu=GPU_CONFIG,  # GPU must be present to load tensorrt_llm
)

MAX_INPUT_LEN, MAX_OUTPUT_LEN = 256, 256
MAX_BATCH_SIZE = 128  # better throughput at larger batch sizes, limited by GPU RAM
ENGINE_DIR = "/root/model/model_output"

SIZE_ARGS = f"--max_batch_size={MAX_BATCH_SIZE} --max_input_len={MAX_INPUT_LEN} --max_seq_len={MAX_OUTPUT_LEN}"

PLUGIN_ARGS = "--paged_kv_cache disable --gemm_plugin auto"

tensorrt_image = tensorrt_image.run_commands(  # update the image by building the TensorRT engine  # takes ~5 minutes
    [
        f"trtllm-build --checkpoint_dir {CKPT_DIR} --output_dir {ENGINE_DIR}"
        + f" {SIZE_ARGS}"
        + f" {PLUGIN_ARGS}"
    ],
    gpu=GPU_CONFIG,  # TRT-LLM compilation is GPU-specific, so make sure this matches production!
).env(  # show more log information from the inference engine
    {"TLLM_LOG_LEVEL": "INFO"}
)

app = modal.App(f"example-trtllm-{MODEL_ID.split('/')[-1]}", image=tensorrt_image)


@app.cls(
    gpu=GPU_CONFIG,
    container_idle_timeout=10 * MINUTES,
)
class Model:
    @modal.enter()
    def load(self):
        """Loads the TRT-LLM engine and configures our tokenizer.

        The @enter decorator ensures that it runs only once per container, when it starts.
        """
        import time

        print(
            f"{COLOR['HEADER']}🥶 Cold boot: spinning up TRT-LLM engine{COLOR['ENDC']}"
        )
        self.init_start = time.monotonic_ns()

        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        self.tokenizer = MistralTokenizer.from_file(f"{MODEL_DIR}/tokenizer.model.v3")

        runner_kwargs = dict(
            engine_dir=f"{ENGINE_DIR}",
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),  # this will need to be adjusted to use multiple GPUs
        )

        self.model = ModelRunner.from_dir(**runner_kwargs)

        self.init_duration_s = (time.monotonic_ns() - self.init_start) / 1e9
        print(
            f"{COLOR['HEADER']}🚀 Cold boot finished in {self.init_duration_s}s{COLOR['ENDC']}"
        )

    @modal.method()
    def generate(self, prompts: list[str]):
        """Generate responses to a batch of prompts, optionally with custom inference settings."""
        import time
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from tensorrt_llm.runtime import SamplingConfig

        num_prompts = len(prompts)

        if num_prompts > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {num_prompts} exceeds maximum of {MAX_BATCH_SIZE}"
            )

        print(
            f"{COLOR['HEADER']}🚀 Generating completions for batch of size {num_prompts}...{COLOR['ENDC']}"
        )
        start = time.monotonic_ns()

        parsed_prompts = [
            ChatCompletionRequest(messages=[UserMessage(content=prompt)])
            for prompt in prompts
        ]

        print(
            f"{COLOR['HEADER']}Parsed prompts:{COLOR['ENDC']}",
            *parsed_prompts,
            sep="\n\t",
        )

        inputs_t = [
            self.tokenizer.encode_chat_completion(completion_request).tokens
            for completion_request in parsed_prompts
        ]
        max_len = max(len(tokens) for tokens in inputs_t)
        # 0 is <unk> in mistral v3
        # seems to work ok as padding
        # https://docs.mistral.ai/guides/tokenization/
        # we left-pad cuz that's what folks do these days
        inputs_t = [([0] * (max_len - len(tokens))) + tokens for tokens in inputs_t]

        import torch

        @torch.inference_mode()
        def generate_mamba(encoded_prompts, model, max_tokens, temperature):
            input_ids = [
                torch.tensor(prompt, device="cuda") for prompt in encoded_prompts
            ]

            output = model.generate(
                batch_input_ids=input_ids,
                sampling_config=SamplingConfig(end_id=4, pad_id=0),
            )

            return output.tolist()

        out_tokens = generate_mamba(
            inputs_t, self.model, max_tokens=1024, temperature=0.35
        )

        responses = [
            self.tokenizer.instruct_tokenizer.tokenizer.decode(tkns[0])
            for tkns in out_tokens
        ]

        duration_s = (time.monotonic_ns() - start) / 1e9


        for prompt, response in zip(prompts, responses):
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{prompt}",
                f"\n{COLOR['BLUE']}{response}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01) 

        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated  from {MODEL_ID} in {duration_s:.1f} seconds,"
        )

        return responses


web_image = modal.Image.debian_slim(python_version="3.10")

class GenerateRequest(pydantic.BaseModel):
    prompts: list[str]


@app.function(image=web_image)
@modal.web_endpoint(
    method="POST",
    label=f"{MODEL_ID.lower().replace('.','-').split('/')[-1]}-web",
    docs=True,
)
def generate_web(data: GenerateRequest) -> list[str]:
    """Generate responses to a batch of prompts, optionally with custom inference settings."""
    return Model.generate.remote(data.prompts)

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}
