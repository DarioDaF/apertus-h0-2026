# Takes rules in multi language and executes the segmentation process

from __future__ import annotations
from importlib import import_module

class Segmenter:
    """Language-aware phrase segmenter.

    Usage:
        segmenter = Segmenter("EN")
        units = segmenter("The Federal Council comes into force...")
    """

    def __init__(self, language: str):
        module_name = f".rules_{language.lower()}"

        #this_dir = Path(__file__).resolve().parent
        #if str(this_dir) not in sys.path:
        #    sys.path.insert(0, str(this_dir)) # @TODO: This breaks future loads namespaces, use module name instead or direct file import

        self.rules = import_module(module_name, package=__package__)
        self.nlp = self.rules.build_pipeline()
        self.language = language.upper()

    def __call__(self, text: str) -> list[str]:
        """Segment text into phrase units."""
        doc = self.nlp(text)
        return [span.text for span in doc.spans["phrase_units"]]

    def segment(self, text: str) -> list[str]:
        """Explicit alternative to __call__ (optional)."""
        return self(text)
