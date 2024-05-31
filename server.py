
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from helper_app2024 import get_response


app = FastAPI()

class Question(BaseModel):
    content: str
    sender: str

class Answer(BaseModel):
    content: str
    source: str

@app.get("/")
def beat():
    return "server is working at 8000"


@app.post("/ask")
def read_item( question: Question = None):
    content, _, _ = get_response(question.content, [])
    response = ''
    for item in content:
        if item.choices[0].delta.content is not None:
            response += item.choices[0].delta.content  
    print(content)
    answer = Answer(content=response, source="ChatGPT")
    
    return answer



@app.post("/ask-stream", response_class=StreamingResponse)
def read_item( question: Question = None):
    def iter_response():
        content, _, _ = get_response(question.content, [])
        for item in content:
            if "content" in item.choices[0].delta:
                yield item.choices[0].delta.content
    
    
    return StreamingResponse(iter_response(), media_type='text/event-stream')