import os
from pathlib import Path
from typing import Any, TypedDict, cast
import math

import dotenv
from my_lllm import MyLLLM, Message, my_lllm_init
import gepa.optimize_anything as gepa_oa
#import litellm
#litellm._turn_on_debug()

THIS_FOLDER = Path(__file__).resolve().parent

dotenv.load_dotenv()
my_lllm_init() # DO AFTER LOAD DOTENV

lmTarget = MyLLLM(
    model=os.environ['LLM_NAME'],

    temperature=0.0,
    max_tokens=2048,

    logprobs=True,
    top_logprobs=5,
)
lmReflect = MyLLLM(
    model=os.environ['LLM_REFLECT_NAME'],
)

class MyPromptCandidate(TypedDict):
    prompt_template: str
MyDataSetId = int
class MyInput(TypedDict):
    original: str
    new_prefix: str
    new_suffix: str
class MyInputWithTarget(MyInput):
    new_middle: str

seed_candidate: MyPromptCandidate = {
    # Using 2 options makes the pareto system not work well at all and reflection LLM seems to not understand the constraints of each field, so unified
    #'prompt_template': 'Fill the appropriate word in the translated text so it mantains the same meaning as original. | [ ## original text ## ]\n{original}\n\n[ ## new text prefix ## ]\n{new_prefix}\n\n[ ## new text suffix ## ]\n{new_suffix}\n\nOutput few words to fill the gap.',
    'prompt_template': '|{original}\n\n{new_prefix}\n\n{new_suffix}\n\n',
}
template_vars = ['original', 'new_prefix', 'new_suffix']
def evaluator(candidate: MyPromptCandidate, example: MyInputWithTarget) -> tuple[float, dict[str, Any]]:
    parts = candidate['prompt_template'].split('|', 1)
    if len(parts) != 2:
        return -math.inf, { 'rejected': True, 'reason': f'Template must contain system prompt and user task separated by "|" to be valid' }
    system = parts[0].strip()
    prompt = parts[1].strip()
    for tvar in template_vars:
        if candidate['prompt_template'].count(f'{{{tvar}}}') != 1:
            return -math.inf, { 'rejected': True, 'reason': f'Template variable {{{tvar}}} is missing from the prompt template' }
        prompt = prompt.replace(f'{{{tvar}}}', example[tvar])
        if system.count(f'{{{tvar}}}') != 0:
            # @NOTE: "GPT6 Luna" seems to not understand this at all, does it even get passed in?
            return -math.inf, { 'rejected': True, 'reason': f'"system" part (before "|") must not contain TEMPLATE variables' }

    # Only call the expensive task LLM if the candidate is valid.
    messages: list[Message] = [
        { 'role': 'system', 'content': system, },
        { 'role': 'user', 'content': prompt, },
    ]
    output = lmTarget.completion(messages)
    resText = output.choices[0].message.content or ''

    score = 1.0 if resText == example['new_middle'] else 0

    return score, {
        'rejected': False,
        'output': resText,
        'score_normalized': score,
    }

dataset: list[MyInputWithTarget] = [
    MyInputWithTarget(
        original=f'After the number {eid} comes {eid+1}',
        new_prefix=f'Dopo il numero ',
        new_suffix=f' viene {eid+1}',
        new_middle=f'{eid}',
    )
    for eid in range(27, 34)
]
valset: list[MyInputWithTarget]|None = [
    MyInputWithTarget(
        original=f'After the number {eid} comes {eid+1}',
        new_prefix=f'Dopo il numero ',
        new_suffix=f' viene {eid+1}',
        new_middle=f'{eid}',
    )
    for eid in range(12, 18)
]

res = gepa_oa.optimize_anything(
    seed_candidate=cast(dict[str, str], seed_candidate),
    evaluator=evaluator,
    dataset=dataset,
    valset=valset,
    objective='From a base text in one language, find the correct text to fill the gap between prefix and suffix in the translated version to preserve meaning',
    background='"prompt_template" is a template string and must contain 2 parts split by "|" character, the first is the system prompt, the second must contain the "{original}", "{new_prefix}", and "{new_suffix}" placeholders that will be filled with the question parts',
    config=gepa_oa.GEPAConfig(
        engine=gepa_oa.EngineConfig(
            max_metric_calls=50,
        ),
        reflection=gepa_oa.ReflectionConfig(
            reflection_lm=lmReflect,
        ),
    ),
)

(THIS_FOLDER / 'artifacts' / 'raw_gepa.candidates.html').write_text(res.candidate_tree_html(), encoding='utf-8')

print(res.best_candidate)
