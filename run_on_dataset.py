from functools import cache
import os
from pathlib import Path
import csv
import re
import random
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import dotenv
from my_lllm import Message, MyLLLM, my_lllm_init
from assistent_prob import AssistentProb_ChatParams, getAssistentProb

THIS_FOLDER = Path(__file__).resolve().parent


RNG_SEED = b'\xA4\xAA\x86\xEA\x17\xFC\x20\xDC'
OUT_FILE = Path(__file__).resolve().with_suffix('.csv')
ENTRIES_COUNT = 160
HOLES_PER_ENTRY = 15


RE_ANY_LETTER = re.compile('[a-zA-Z]')

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
from datasets import load_dataset
def dataitem2example(item):
    return {
        'id': item['id'], 'lang_A_id': 'en', 'lang_B_id': item['subset'].split('_')[1].upper(),
        'text_A': item['text_a_untokenized'], 'text_B': item['text_b_untokenized'],
    }
examples = [ dataitem2example(item) for item in load_dataset('ZurichNLP/SwissGov-RSD', split='validation') ]
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

    with ThreadPoolExecutor(max_workers=5) as executor:
        def process_entry(example):
            print('Next example...')
            ress = []
            for _ in range(HOLES_PER_ENTRY):
                hole_src, other_src = seeded_random.choice([ ('A', 'B'), ('B', 'A') ])

                print(example['id'])

                segmenter = get_segmenter(example[f'lang_{hole_src}_id'])
                seg_text = segmenter.segment(example[f'text_{hole_src}'])
                other_text = example[f'text_{other_src}']

                valid_hole_idxs = [ idx for idx, hole_text in enumerate(seg_text) if RE_ANY_LETTER.match(hole_text) is not None ]

                hole_idx = seeded_random.choice(valid_hole_idxs)

                myin = MyInputWithTarget(
                    original=other_text,
                    new_prefix=' '.join(seg_text[:hole_idx]) + ' ',
                    new_middle=seg_text[hole_idx],
                    new_suffix=' ' + ' '.join(seg_text[hole_idx+1:]),
                )

                assistant_lprob, _ = getAssistentProb(lmTarget, genPrompt(myin))

                ress.append({
                    'id': example['id'],
                    'hole_src': hole_src,
                    'hole_idx': hole_idx,
                    'assistant_lprob': assistant_lprob,
                })
            return ress


        futures = [
            executor.submit(process_entry, entry)
            for entry in examples[:ENTRIES_COUNT]
        ]

        for future in as_completed(futures):
            values = future.result()
            w.writerows(values)
            fp.flush()
