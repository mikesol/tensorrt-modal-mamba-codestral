
# tensorrt-modal-mamba-codestral

Adapted from the [modal llama3 example](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving).

Mistral doesn't really play nice with huggingface, so we use their APIs from `mistral-inference`.

## To run

```bash
modal serve main.py
```