
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from db import update_chat_history
from helper_app2024 import get_response
from models import Answer, Question, ResponsePair


app = FastAPI()



@app.get("/")
def beat():
    return "server is working at 8000"


@app.post("/ask")
def response_for_query( question: Question = None):
    content, _, _ = get_response(question.content, [])
    response = ''
    for item in content:
        if item.choices[0].delta.content is not None:
            response += item.choices[0].delta.content 
    
    answer = Answer(content=response, source="ChatGPT")
    qa_pair = ResponsePair(question=question.content, answer= answer.content)
    
    update_chat_history(question, qa_pair)
    
    return answer



@app.post("/ask-stream", response_class=StreamingResponse)
def response_streaming_for_query( question: Question = None):
    def iter_response():
        content, _, _ = get_response(question.content, [])
        for item in content:
            if "content" in item.choices[0].delta:
                yield item.choices[0].delta.content
    
    
    return StreamingResponse(iter_response(), media_type='text/event-stream')