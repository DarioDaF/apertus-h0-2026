import os
from pathlib import Path
from typing import TypedDict
import math

import dotenv
from my_lllm import MyLLLM, Message, my_lllm_init
from assistent_prob import getAssistentProb

THIS_FOLDER = Path(__file__).resolve().parent

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'
def prettyPrintMessage(m: Message):
    colors = { 'system': GREEN, 'user': BLUE, 'assistant': RED }
    print(f'{m['role']}: {colors.get(m['role'], YELLOW)}{m['content']}{RESET}')

dotenv.load_dotenv()
my_lllm_init() # DO AFTER LOAD DOTENV

lmTarget = MyLLLM(
    model=os.environ['LLM_NAME'],

    temperature=0.0,
    max_tokens=30,
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


example = MyInputWithTarget(
    original='Legislation Administration The Federal Chancellery is responsible for legislation regulating the procedures of the government and the Federal. This includes laws on the consultation procedure, publications, government and administrative organisation and on Parliament. The Federal Chancellery drafts and enforces laws in these areas. For full information please select German. Links Sektion Recht Specialist staff',
    new_prefix='Legislazione La Cancelleria federale è competente per le leggi che disciplinano l’organizzazione del Governo e dell’Amministrazione, ossia il diritto in materia di organizzazione dell’Amministrazione, il diritto in materia di procedura di consultazione, il diritto in materia di pubblicazioni ufficiali e il diritto parlamentare. In tali ambiti la Cancelleria federale prepara gli ',
    new_suffix=' e li esegue. Links Sezione del diritto Messaggio per lo specialista',
    new_middle='atti legislativi',
)
#example = MyInputWithTarget(
#    original='Il gatto passeggiava sulla riva del fiume',
#    new_prefix='On the river bank ',
#    new_suffix=' was walking',
#    new_middle='a cat',
#)

prompt = genPrompt(example)
print('\n\nPROMPT:')
for m in prompt:
    prettyPrintMessage(m)

print('\n\nBEST MODEL RESPONSE:')
prettyPrintMessage(lmTarget.completion(prompt[:-1]).choices[0].message)

lprob, tkchoices = getAssistentProb(lmTarget, genPrompt(example))

print('\n\nACTUAL TEXT INTO THE MODEL:')
print(YELLOW, end='')
for tkchoice in tkchoices:
    print(tkchoice['token_probs'][tkchoice['token']]['decoded_token'], end='')
print(RESET)

print(f'\n\nProbability: {math.exp(lprob)*100:0.6f} % [{lprob:0.2f} dB]')
