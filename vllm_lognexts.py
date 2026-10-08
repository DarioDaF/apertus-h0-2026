import os
from pathlib import Path
import math

import dotenv
from my_lllm import MyLLLM, my_lllm_init
from assistent_prob import AssistentProb_ChatParams, getAssistentProb

THIS_FOLDER = Path(__file__).resolve().parent

dotenv.load_dotenv()
my_lllm_init() # DO AFTER LOAD DOTENV

lmTarget = MyLLLM(
    model=os.environ['LLM_NAME'],

    **AssistentProb_ChatParams,
)

assistant_lprob, tkchoices = getAssistentProb(lmTarget, [
    { 'role': 'user', 'content': 'Tell me the words in a random language: Hello world' },
    { 'role': 'assistant', 'content': 'Hello world' },
])

print('<style>:root { font-family: sans-serif; }</style>')
i = 0
for tkchoice in tkchoices:
    best_tk, best_params = [ (k, v) for k, v in tkchoice['token_probs'].items() if v['rank'] == 1 ][0]
    tkprobs = tkchoice['token_probs'][tkchoice['token']]
    #
    rank = tkprobs['rank']
    compressed_rank = math.atan(rank-1)/3.2
    tkprob_lin = math.exp(tkprobs['logprob'])
    best_text = best_params['decoded_token']
    text = tkprobs['decoded_token']
    print(f'<span style="background-color: #FFFF{'FF' if i%2==0 else 'C0'}; color: #0000{int(255*compressed_rank):02X}" title="[{tkprob_lin*100:0.2f} % | {rank}] (best: {best_text})">{text}</span>', end='')
    #
    i += 1

print(f'\n<br /><br /><p>Assistant prob: {math.exp(assistant_lprob)*100:0.2f} %</p>')
