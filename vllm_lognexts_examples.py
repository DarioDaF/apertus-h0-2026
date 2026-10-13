from functools import cache
import os
from pathlib import Path
import csv
import random
from typing import TypedDict

import dotenv
from my_lllm import Message, MyLLLM, my_lllm_init
from assistent_prob import AssistentProb_ChatParams, getAssistentProb

THIS_FOLDER = Path(__file__).resolve().parent


RNG_SEED = b'\xA4\xAA\x86\xEA\x17\xFC\x20\xDC'
IN_FILE = THIS_FOLDER / 'examples_clean.csv'
OUT_FILE = Path(__file__).resolve().with_suffix('.csv')
ENTRIES_COUNT = 160
HOLES_PER_ENTRY = 4


dotenv.load_dotenv()
my_lllm_init() # DO AFTER LOAD DOTENV

seeded_random = random.Random(RNG_SEED)

lmTarget = MyLLLM(
    model=os.environ['LLM_NAME'],

    **AssistentProb_ChatParams,
)

class MyInput(TypedDict):
    original: str
    new_prefix: str
    new_suffix: str
class MyInputWithTarget(MyInput):
    new_middle: str

def genPrompt(x: MyInputWithTarget) -> list[Message]:
    return [
        { 'role': 'system', 'content': 'Fill the appropriate part of speach in the translated text so it mantains the same meaning as original. Do not output any explaination or text or extra content except the required filling words', },
        { 'role': 'user', 'content': f'[ ## original text ## ]\n{x['original']}\n\n[ ## new text prefix ## ]\n{x['new_prefix']}\n\n[ ## new text suffix ## ]\n{x['new_suffix']}', },
        { 'role': 'assistant', 'content': f'{x['new_middle']}', },
    ]

@cache
def get_segmenter(lang: str):
    return Segmenter(lang)


### Load phrase_segmenter
print('Loading "phrase_segmenter"...')
from phrase_segmenter import Segmenter

### Load examples
print('Loading examples...')
with IN_FILE.open('r', encoding='utf-8') as fp:
    examples = [*csv.DictReader(fp)]
print(f'Loaded {len(examples)} entries from example table')
seeded_random.shuffle(examples)

### Compute
print('Compute...')
with OUT_FILE.open('w', encoding='utf-8') as fp:
    w = csv.DictWriter(fp, [
        'id', 'hole_src', 'hole_idx',
        'assistant_lprob', 'assistant_rank', 'assistant_preferred',
        'original', 'new_prefix', 'new_middle', 'new_suffix',
    ], lineterminator='\n')
    w.writeheader()

    for example in examples[:ENTRIES_COUNT]:
        print('Next example...')
        for _ in range(HOLES_PER_ENTRY):
            hole_src, other_src = seeded_random.choice([ ('A', 'B'), ('B', 'A') ])

            print(example['id'])

            segmenter = get_segmenter(example[f'lang_{hole_src}_id'])
            seg_text = segmenter.segment(example[f'text_{hole_src}'])
            other_text = example[f'text_{other_src}']

            hole_idx = seeded_random.randrange(0, len(seg_text))

            myin = MyInputWithTarget(
                original=other_text,
                new_prefix=' '.join(seg_text[:hole_idx]) + ' ',
                new_middle=seg_text[hole_idx],
                new_suffix=' ' + ' '.join(seg_text[hole_idx+1:]),
            )

            '''
            assistant_lprob, tkchoices = getAssistentProb(lmTarget, genPrompt(myin))

            rank = -1
            best_text = '<|INVALID|>'
            for tkchoice in reversed(tkchoices):
                tkprobs = tkchoice['token_probs'][tkchoice['token']]
                if tkprobs['decoded_token'] == '<|assistant_start|>':
                    break
                best_tk, best_params = [ (k, v) for k, v in tkchoice['token_probs'].items() if v['rank'] == 1 ][0]
                tkprobs = tkchoice['token_probs'][tkchoice['token']]
                rank = tkprobs['rank']
                tkprob_logprob = tkprobs['logprob']
                best_text = best_params['decoded_token']

            w.writerow({
                **myin,
                'id': example['id'],
                'hole_src': hole_src,
                'hole_idx': hole_idx,
                'assistant_lprob': assistant_lprob,
                'assistant_rank': rank,
                'assistant_preferred': best_text,
            })
            fp.flush()
            '''
