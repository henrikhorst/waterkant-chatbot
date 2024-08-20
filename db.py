from pydantic import BaseModel
from pymongo import MongoClient

from models import ChatHistory, Question, ResponsePair


client = MongoClient('mongodb://localhost:27017/')
db = client.waterkant_bot
user_chat_history_collection = db.user_chat_history

def upsert_user_chat_history(chat_history: ChatHistory) -> ChatHistory:
    query = {"userId": chat_history.userId}
    data_to_patch = {"$set": chat_history.model_dump(by_alias=True)}
    
    updated_chat_history = user_chat_history_collection.update_one(query, data_to_patch, upsert=True)
    return updated_chat_history

def get_chat_history(user_id: str)-> ChatHistory | None:
    
    chat_history = user_chat_history_collection.find_one({"userId": user_id})
    
    if chat_history:
        return ChatHistory(**chat_history)
    return None


def update_chat_history(userQuestion: Question, response: ResponsePair) -> ChatHistory :
    user_chat_history = get_chat_history(userQuestion.userId)
    
    if user_chat_history is None:
        user_chat_history = ChatHistory(userId=userQuestion.userId, messages=[])
    
    user_chat_history.messages.append(response)
    upserted = upsert_user_chat_history(user_chat_history)
    return upserted
    