from pydantic import BaseModel

class TextInput(BaseModel):
    text: str
    max_keywords: int = 5

class KeywordOutput(BaseModel):
    keywords: list[str]
    probabilities: list[float]