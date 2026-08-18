from core.extractor import BaseExtractor
from core.prompts import EXPERIMENTAL_DESIGN_PROMPT

class ExperimentalDesignAgent(BaseExtractor):
    def get_prompt(self, text: str) -> str:
        return EXPERIMENTAL_DESIGN_PROMPT.format(descriptor=text)
