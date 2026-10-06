This repository expects [`uv`](https://docs.astral.sh/uv/) package manager to be used to run the programs (`uv run file.py`).

Environment file `.env` expects these fields filled in:
- `LLM_NAME`: LLM model `litellm` style name
- `LLM_REFLECT_NAME`: LLM model `litellm` style name for the **gepa** reflection
- `LLM_APERTUS_BASE_URL`: Url of endpoint to call `apertus/*` model from
- `LLM_APERTUS_API_KEY`: Api key for the endpoint of `apertus/*`

LiteLLM style models expect the format `provider/model_name` and in this case `apertus` endpoint was registered as a provider (as long as `my_lllm.MyLLLM` is used to call them).

LiteLLM for `chatgpt/*` provider is bugged and https://github.com/BerriAI/litellm/pull/41235 fixes the streaming override issue, so the version with the fix is pinned until there is an official release after the merge
