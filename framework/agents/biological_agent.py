from core.extractor import BaseExtractor
from core.prompts import BIOLOGICAL_PROMPT

class BiologicalAgent(BaseExtractor):
    def get_prompt(self, text: str) -> str:
        return BIOLOGICAL_PROMPT.format(descriptor=text)
