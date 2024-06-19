import json
import uuid
import numpy as np
from openai import OpenAI
import streamlit as st
import helper
from helper import find_k_nearest_neighbors
from helper import waterkant_festival_description
from helper import Document
from helper import replace_placeholders
from datetime import datetime

from utils import get_api_secret

def filter_instances(instances, filters):
    filtered_instances = []
    
    for instance in instances:
        match = True
        # Check if both time constraints are given
        both_times = 'start_time' in filters and 'end_time' in filters

        for key, value in filters.items():
            if key == 'Speaker':
                # 'Speaker' filter should match any speaker in the list
                if value not in instance.metadata['Speakers']:
                    match = False
                    break
            elif key in instance.metadata['date']:
                if key == 'day' and instance.metadata['date'][key] != value:
                    match = False
                    break
                    
                # Handling 'start_time' and 'end_time' specifically
                elif both_times:
                    # When both start and end times are provided
                    if key == 'start_time':
                        if instance.metadata['date'][key] < value or instance.metadata['date']['end_time'] > filters['end_time']:
                            match = False
                            break
                    elif key == 'end_time':
                        if instance.metadata['date'][key] > value or instance.metadata['date']['start_time'] < filters['start_time']:
                            match = False
                            break
                else:
                    # Handle individual time constraints
                    if key == 'start_time' and instance.metadata['date'][key] < value:
                        match = False
                        break
                    elif key == 'end_time' and instance.metadata['date'][key] > value:
                        match = False
                        break
            else:
                # If the filter key doesn't exist, ignore that filter
                continue

        if match:
            filtered_instances.append(instance)
    
    return filtered_instances

events_template="**List of events**\n\n%events%\n\nPlease provide an answer to the question based on the preceding information. Include all events that directly relate to the topic of the question. Additionally, mention any events that, while not directly related to the specified time, could still be relevant and interesting. Exclude any events that do not pertain to the topic. Answer in the same language as the question.Here is the question:\n%question%"

np_embeddings = np.load('prep/2024/earlyembeddings.npy')

k=len(np_embeddings)

with open('prep/2024/documents.json', 'r') as file:
            json_documents = json.load(file)

loaded_documents = [Document.from_json(json_str) for json_str in json_documents]

key = get_api_secret()
client = OpenAI(api_key=key )

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieves_events_by_speaker_and_day_and_time_constraints",
            "description": "Retrieves a list of events fitting to the question. If specified in the question it filters the events also based on speaker, specific day and earliest and latest permissible times for event start and end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "interest_description": {
                        "type": "string",
                        "description": "Verbal formulation from the question what is the interest",
                    },
                    "day": {
                        "type": "string",
                        "description": "The specific days of the week to filter events for, e.g. 'Monday', 'Friday' or 'Tuesday'"
                    },
                    "min_start_time": {
                        "type": "string",
                        "description": "The earliest permissible start time for events, in 24-hour format, e.g., '08:00' for 8 AM."
                    },
                    "max_end_time": {
                        "type": "string",
                        "description": "The latest permissible ending time for events, in 24-hour format, e.g., '17:00' for 5 PM"
                    },
                    "speaker": {
                        "type": "string",
                        "description": "The specific speaker to filter events for, e.g. 'Henrik Horst', 'Steffen Brandt' or 'Dirk Schrödter'"
                    },
                    
                },
                "required": ["interest_description"],
            }
        }
    },
]

GPT_MODEL = "gpt-3.5-turbo"
def chat_completion_request(messages, tools=None, model=GPT_MODEL, stream=False):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature = 0.0,
            stream=stream
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

system_message = waterkant_festival_description+"You are a helpful chatbot for the attendees of the Waterkant Festival! You answer questions about events, schedules, venue information, transportation options, or anything else. Make no speculations what topics or events are on the festival but call the function then instead. If you are asked about any topic, event or something that is not directly be answered in a meaningful way by the description above, call the retrieve_events_by_speaker_and_day_and_time_constraints function instead if speaker (e.g. 'Dirk Schrödter') and/or time constraints are mentioned e.g. 'Friday morning'. You respond in the same language the user asks the question!"


def get_response(question, messages):
    if not messages:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": question})
    chat_response = chat_completion_request(messages, tools)
    assistant_message = chat_response.choices[0].message
    if assistant_message.tool_calls:
        assistant_message.content = str(assistant_message.tool_calls[0].function)
        messages.append({"role": assistant_message.role, "content": assistant_message.content})
        filters = json.loads(assistant_message.tool_calls[0].function.arguments)
        eventfilters = {}
        for key in filters.keys():
            if key=="speaker" and filters[key]:
                eventfilters["Speaker"]= filters[key]
            elif key=="day" and filters[key]:
                eventfilters["day"]= filters[key]
            elif key=="min_start_time" and filters[key]:
                eventfilters["start_time"]= filters[key]
            elif key=="max_end_time"  and filters[key]:
                eventfilters["end_time"]= filters[key]
        filtered_events = filter_instances(loaded_documents, eventfilters)
        filtered_events_set = set(filtered_events)  # Convert to set for faster lookup
        # Convert question to embedding
        question_embedding = get_embedding(filters["interest_description"])
        np_question_embedding = np.array(question_embedding)

        # Find k nearest neighbors
        indices, similarities = find_k_nearest_neighbors(
            np_embeddings, np_question_embedding, k
        )
        picked_events = []
        for indice in indices:
            if loaded_documents[indice] in filtered_events_set:
                picked_events.append(loaded_documents[indice])
                if len(picked_events) >= 8:
                    break
        if not picked_events:
            events_str = "No events found for the request"
        else:
            events_str = ""
            for event in picked_events:
                events_str += event.content + "\n\n"
        messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": events_str})
        replacements= {"question": question, "events": events_str}
        prompt= replace_placeholders(events_template, replacements)
        events_messages= [{"role": "system", "content": "You are a helpful chatbot of the Waterkantfestival with answers question in an engaging way using always a few smileys. Very importantly you answer always in the langaguage of the question no matter the language of the instructions"},{"role": "user", "content": prompt}]
        response = chat_completion_request(events_messages, tools=None, model=GPT_MODEL, stream=True)
        
        #messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response, messages, True

    else:
        messages.append({"role": assistant_message.role, "content": assistant_message.content})
        return assistant_message.content, messages, False

    

