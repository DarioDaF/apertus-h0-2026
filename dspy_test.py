import os, json
from pathlib import Path
from typing import TypedDict

import dotenv
import dspy

THIS_FOLDER = Path(__file__).resolve().parent

dotenv.load_dotenv()

lmTarget = dspy.LM(
    model=os.environ['LLM_NAME'].replace('apertus/', 'openai/'),
    api_base=os.environ['LLM_APERTUS_BASE_URL'],
    api_key=os.environ['LLM_APERTUS_API_KEY'],

    temperature=0.0,
    max_tokens=2048,

    logprobs=True,
    top_logprobs=5,
)
lmReflect = dspy.LM(
    model=os.environ['LLM_REFLECT_NAME'].replace('apertus/', 'openai/'),
    api_base=os.environ['LLM_APERTUS_BASE_URL'],
    api_key=os.environ['LLM_APERTUS_API_KEY'],
)

class FillTranslation(dspy.Signature):
    """Fill the appropriate word in the translated text so it mantains the same meaning as original."""
    original: str = dspy.InputField()
    translated_prefix: str = dspy.InputField()
    translated_suffix: str = dspy.InputField()
    translated_middle: str = dspy.OutputField()

pipeline = dspy.Predict(FillTranslation)
dspy.configure(
    lm=lmTarget,
    adapter=dspy.JSONAdapter(),
)
#res = pipeline(
#    original='Il gatto passeggiava sulla riva del fiume',
#    translated_prefix='On the river bank ',
#    translated_suffix=' was walking'
#)
res = pipeline(
    original='Legislation Administration The Federal Chancellery is responsible for legislation regulating the procedures of the government and the Federal. This includes laws on the consultation procedure, publications, government and administrative organisation and on Parliament. The Federal Chancellery drafts and enforces laws in these areas. For full information please select German. Links Sektion Recht Specialist staff',
    translated_prefix='Legislazione La Cancelleria federale è competente per le leggi che disciplinano l’organizzazione del Governo e dell’Amministrazione, ossia il diritto in materia di organizzazione dell’Amministrazione, il diritto in materia di procedura di consultazione, il diritto in materia di pubblicazioni ufficiali e il diritto parlamentare. In tali ambiti la Cancelleria federale prepara gli ',
    translated_suffix=' e li esegue. Links Sezione del diritto Messaggio per lo specialista'
)
(THIS_FOLDER / 'probs.json').write_text(json.dumps(res.logprobs, indent=2), encoding='utf-8')

class LogProb_Token(TypedDict):
    token: str
    lobprob: float
class LogProb_Choice(LogProb_Token):
    top_logprobs: list[LogProb_Token]
class LogProbs(TypedDict):
    content: list[LogProb_Choice]

logprobs: LogProbs = res.logprobs

print(res.translated_middle)
