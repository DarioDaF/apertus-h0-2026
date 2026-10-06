import os
from pathlib import Path
from typing import Any, TypedDict, cast
import math

import dotenv
import gepa.optimize_anything as gepa_oa
#import litellm
#litellm._turn_on_debug()

THIS_FOLDER = Path(__file__).resolve().parent

dotenv.load_dotenv()

from my_lllm import MyLLLM, Message # DO AFTER LOAD DOTENV

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
    system: str
    prompt_template: str
MyDataSetId = int
class MyInput(TypedDict):
    original: str
    new_prefix: str
    new_suffix: str
class MyInputWithTarget(MyInput):
    new_middle: str

seed_candidate: MyPromptCandidate = {
    #'system': 'Fill the appropriate word in the translated text so it mantains the same meaning as original.',
    #'prompt_template': '[ ## original text ## ]\n{original}\n\n[ ## new text prefix ## ]\n{new_prefix}\n\n[ ## new text suffix ## ]\n{new_suffix}\n\nOutput few words to fill the gap.',
    'system': '',
    'prompt_template': '{original}\n\n{new_prefix}\n\n{new_suffix}\n\n',
}
template_vars = ['original', 'new_prefix', 'new_suffix']
def evaluator(candidate: MyPromptCandidate, example: MyInputWithTarget) -> tuple[float, dict[str, Any]]:
    prompt = candidate['prompt_template']
    for tvar in template_vars:
        if candidate['prompt_template'].count(f'{{{tvar}}}') != 1:
            return -math.inf, { 'rejected': True, 'reason': f'Template variable {{{tvar}}} is missing from the prompt template' }
        prompt = prompt.replace(f'{{{tvar}}}', example[tvar])
        if candidate['system'].count(f'{{{tvar}}}') != 0:
            # @NOTE: "GPT6 Luna" seems to not understand this at all, does it even get passed in?
            return -math.inf, { 'rejected': True, 'reason': f'"system" must not contain TEMPLATE variables' }

    # Only call the expensive task LLM if the candidate is valid.
    messages: list[Message] = [
        { 'role': 'system', 'content': candidate['system'], },
        { 'role': 'user', 'content': prompt, },
    ]
    output = lmTarget.completion(messages)
    resText = output.choices[0].message.content or ''

    score = 1.0 if resText == example['new_middle'] else 0

    return score, {
        'rejected': False,
        'output': resText,
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
    background='"prompt_template" is a template string and must contain the "{original}", "{new_prefix}", and "{new_suffix}" placeholders that will be filled with the question parts, all the other parts are NOT TEMPLATES',
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
