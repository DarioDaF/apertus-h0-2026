'''
    Description: LiteLLM wrapper to conform to `gepa.LanguageModel` but allows also raw requests and limits
    Author: DarioDaF
'''

from litellm import ModelResponse, Choices
from litellm.router import Router
from litellm.types.router import Deployment, LiteLLM_Params
import os
from typing import TypedDict, Literal, Any
from my_limiter import MyLimiter, myNoopLimiter

from logging import Logger

lllm_logger = Logger(__name__)

providersLimiter: dict[str, MyLimiter] = {}
lllm_router = Router(
    model_list=[
        # Allow all default providers passthrough
        {
            "model_name": "*",
            "litellm_params": {
                "model": "*",
            },
        },
    ]
)

def my_lllm_init():
    providersLimiter['apertus'] = MyLimiter(min_delay_between=0.2, max_concurrent=5)
    lllm_router.add_deployment(Deployment(
        model_name='apertus/*',
        litellm_params=LiteLLM_Params(
            model='openai/*',
            api_base=os.environ['LLM_APERTUS_BASE_URL'],
            api_key=os.environ['LLM_APERTUS_API_KEY'],

            # All models on this endpoint are free
            input_cost_per_token=0,
            output_cost_per_token=0,
            cache_creation_input_token_cost=0,
            cache_read_input_token_cost=0,
        )
    ))

class Message(TypedDict):
    role: Literal['system', 'user', 'assistant']
    content: str

class MyLLLM:
    def __init__(self, **kwargs) -> None:
        self.model = kwargs['model']
        del kwargs['model']
        self.kwargs = kwargs
    def _check_truncation(self, choices: list[Choices], combined_kwargs: dict[str, Any]) -> None:
        if any(getattr(c, "finish_reason", None) == "length" for c in choices):
            max_tok = combined_kwargs.get("max_tokens") or self.kwargs.get("max_completion_tokens")
            lllm_logger.warning(
                f"LM response was truncated (finish_reason='length', max_tokens={max_tok}). "
                "Consider increasing max_tokens for better results."
            )
    def completion(self, messages: list[Message], **kwargs) -> ModelResponse:
        global lllm_router, providersLimiter
        limiter = providersLimiter.get(self.model.split('/')[0], myNoopLimiter)
        combined_kwargs = { **self.kwargs, **kwargs }
        with limiter.acquire():
            completion: ModelResponse = lllm_router.completion(model=self.model, messages=messages, **combined_kwargs) # type: ignore[union-attr]
        self._check_truncation(completion.choices, combined_kwargs)
        return completion
    async def acompletion(self, messages: list[Message], **kwargs) -> ModelResponse:
        global lllm_router, providersLimiter
        limiter = providersLimiter.get(self.model.split('/')[0], myNoopLimiter)
        combined_kwargs = { **self.kwargs, **kwargs }
        async with limiter.aacquire():
            completion: ModelResponse = await lllm_router.acompletion(model=self.model, messages=messages, **combined_kwargs) # type: ignore[union-attr]
        self._check_truncation(completion.choices, combined_kwargs)
        return completion
    def __call__(self, prompt: str|list[dict[str, str]], **kwargs) -> str:
        messages: list[Message]
        if isinstance(prompt, str):
            messages = [ { 'role': 'user', 'content': prompt } ]
        else:
            messages = prompt # type: ignore[union-attr]
        completion = self.completion(messages, **kwargs)
        return completion.choices[0].message.content # type: ignore[union-attr]
