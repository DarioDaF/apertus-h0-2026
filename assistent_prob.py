from typing import TypedDict
from my_lllm import MyLLLM, Message
from dataclasses import dataclass

AssistentProb_ChatParams = {
    'temperature': 0.0,

    'logprobs': False, # used for continuation choosen probs, not prompt ones
    'top_logprobs': 0, # 5 is max, used for continuation choosen probs, not prompt ones

    'return_token_ids': True,
    'continue_final_message': False, # True if you want further generation and no <|assistent_end|> token
    'add_generation_prompt': False,
    'prompt_logprobs': 5,
    'max_tokens': 1, # Can't be 0 on chat endpoint
}

### For text_completion endpoint would be
#{
#  "model": "your-model",
#  "prompt": "<formatted chat prompt including assistant prefill>",
#  "echo": true,
#  "max_tokens": 0,
#  "logprobs": 5 # ??? not prompt_lobprobs? would need testing
#}

class TokenPorb(TypedDict):
    rank: int
    logprob: float
    decoded_token: str
class TokenChoice(TypedDict):
    token: str
    token_probs: dict[str, TokenPorb]

tuple[str, dict[str, TokenPorb]]

def getTokenPromptProbChoices(llm: MyLLLM, messages: list[Message]) -> list[TokenChoice]:
    global AssistentProb_ChatParams
    resp = llm.completion(messages, **AssistentProb_ChatParams) # @NOTE: I know for now it warns of truncation very likely even tho it works
    res: list[TokenChoice] = []
    for plps, tk in zip(resp.prompt_logprobs, resp.prompt_token_ids, strict=True):
        if plps is None:
            continue
        res.append({
            'token': str(tk),
            'token_probs': plps
        })
    return res

def getAssistentProb(llm: MyLLLM, messages: list[Message], include_end = True):
    assert(messages[-1]['role'] == 'assistant') # Last message must be assistent
    resp = getTokenPromptProbChoices(llm, messages)

    assistant_lprob = 0.0
    for tkchoice in reversed(resp):
        tkprobs = tkchoice['token_probs'][tkchoice['token']]
        if tkprobs['decoded_token'] == '<|assistant_start|>':
            break
        if not include_end and (tkprobs['decoded_token'] == '<|assistant_end|>'):
            continue
        assistant_lprob += tkprobs['logprob']

    return assistant_lprob, resp
