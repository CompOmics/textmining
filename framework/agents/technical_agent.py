from core.extractor import BaseExtractor
from core.prompts import TECHNICAL_PROMPT

class TechnicalAgent(BaseExtractor):
    def get_prompt(self, text: str) -> str:
        return TECHNICAL_PROMPT.format(descriptor=text)
