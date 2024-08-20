from typing import List
from pydantic import BaseModel


class ResponsePair(BaseModel):
    question: str
    answer: str

class Question(BaseModel):
    content: str
    sender: str
    userId: str

class Answer(BaseModel):
    content: str
    source: str

class ChatHistory(BaseModel):
    messages: List[ResponsePair]
    userId: str