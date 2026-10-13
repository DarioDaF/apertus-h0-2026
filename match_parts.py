from functools import cache
import os
from pathlib import Path
import csv
import re
import random
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import dotenv

THIS_FOLDER = Path(__file__).resolve().parent


RE_SPLI_LOGIC = re.compile(r'(?:[^\W_]+)|(?:[^ \r\n\t\xA0])') # Include nbsp

RE_ANY_LETTER = re.compile('[a-zA-Z]')
RE_ONLY_WS = re.compile(r'^[ \r\n\t]*$')

dotenv.load_dotenv()

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
        'labels_A': item['labels_a'], 'labels_B': item['labels_b'],
        'tk_A': item['text_a'],
    }
examples = [ dataitem2example(item) for item in load_dataset('ZurichNLP/SwissGov-RSD', split='validation') ]
print(f'Loaded {len(examples)} entries from example table')

example = examples[0]
#items = [ e[0] for e in RE_SPLI_LOGIC.finditer(example['text_A']) ]
#other_items = example['tk_A'].split(' ')
#for x, y in zip(items, other_items):
#    print(x, y)
#    if x != y:
#        break
for example in examples:
    other_items = example['tk_A'].split(' ')
    assert len(example['labels_A']) == len(other_items)
'''
parts = get_segmenter('EN').segment(example['text_A'])
i = 0
for part in parts:
    example['text_A'].split('')
    if RE_ONLY_WS.match(part):
        continue
    if not RE_ANY_LETTER.match(part):
        assert example['labels_A'][i] == -1
    i += len(part.split(' '))
'''
