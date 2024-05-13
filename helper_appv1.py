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

def filter_events(events, filters):
    filtered_events = []
    # Unpacking the filter criteria
    days = filters['days']
    min_start_time_str = filters['min_start_time']
    max_end_time_str = filters['max_end_time']
    
    # Safely parse the global min and max times
    try:
        min_start_time = datetime.strptime(min_start_time_str, "%H:%M").time()
        max_end_time = datetime.strptime(max_end_time_str, "%H:%M").time()
    except ValueError:
        print("Global time filters are not in the correct format (HH:MM). No events will be processed.")
        return filtered_events

    for event in events:
        event_day = event.metadata['date']['day']
        # Safely parse event start and end times
        try:
            event_start_time = datetime.strptime(event.metadata['date']['start_time'], "%H:%M").time()
            event_end_time = datetime.strptime(event.metadata['date']['end_time'], "%H:%M").time()
        except ValueError:
            #print(f"Skipping event due to incorrect time format: {event}")
            continue

        # Check if the event meets all the criteria
        if event_day in days and event_start_time >= min_start_time and event_end_time <= max_end_time:
            filtered_events.append(event)
    
    return filtered_events

time_template="""Extract time-related information such as days of the week, start times, and end times from a question, and format them as specified, follow these steps:

Identify Time Information: Begin by scrutinizing the question to identify mentions of specific days, times, or durations. Key elements to look for include:
Days of the week (e.g., Monday, Tuesday, etc.).
Time expressions (e.g., "9 AM", "1400 hours", "from 9 to 11").
Keywords suggesting time durations (e.g., "morning", "evening", "all day").
List Days of the Week: If any days are mentioned:
Note each day explicitly mentioned. Include all days in a mentioned range (e.g., "Monday to Friday").
Ensure days are fully spelled out (e.g., "Monday", not "Mon") and capitalized.
Determine Start and End Times:
Look for specific times mentioned for starting or ending, noted in either 12-hour or 24-hour format.
Decide if each time refers to a start time or end time based on contextual keywords like "from", "start at", "until", "end by".
Convert times in 12-hour format (AM/PM) into 24-hour format for consistency (e.g., "2 PM" becomes "14:00").
Calculate Minimum Start Time and Maximum End Time:
From all start times identified, choose the earliest as "min_start_time".
From all end times, select the latest as "max_end_time".
Times should be in the "HH:MM" format (e.g., "09:00", "18:00").
Check for Default Values:
If no days are mentioned in the question, default to ['Thursday', 'Friday'].
If no start or end times are provided, use '00:00' as the default "min_start_time" and '23:00' as the "max_end_time".
Format the Output:
Organize the data into a dictionary format with keys: 'days', 'min_start_time', and 'max_end_time'.
Under 'days', include the listed or default days in square brackets.
Assign the respective or default values to 'min_start_time' and 'max_end_time'.
Example:
If the question is "When are late night discussions held?", and no specific days or times are mentioned, your output should look like this:

{
  "days": ["Thursday", "Friday"],
  "min_start_time": "00:00",
  "max_end_time": "23:00"
}
Example:
If the question is "What are the events on Friday starting at 9 AM and ending before 4 PM?", your formatted output should look like this:
{
  "days": ["Friday"],
  "min_start_time": "00:00",
  "max_end_time": "16:00"
}

Here is the question:
%question%
"""

events_template="Here is a list of events which could be helpful to answer the question:\n\n%events%Answer the question based on the information above. Answer in the language of the question. Answer with which events are really fitting to the question and which might not be directly but could also be interesting. Here is the question again:\n%question%"

np_embeddings = np.load('prep/embeddings.npy')

with open("prep/documents.json", "r") as file:
    json_documents = json.load(file)

loaded_documents = [Document.from_json(json_str) for json_str in json_documents]

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

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

system_prompt = waterkant_festival_description+"You are a helpful chatbot for the attendees of the Waterkant Festival! You answer questions about events, schedules, venue information, transportation options, or anything else. Make no speculations what topics or events are on the festival but call the function then instaed. If you are asked about any topic, event or something that is not directly be answered in a meaningful way by the description above, call the filter_events_by_day_and_time_constraints function. You respond in the same language the user asks the question!"

tools = [{
    "type": "function",
    "function": {
        "name": "filter_events_by_day_and_time_constraints",
        "description": "Filters a list of events based on specific day and time constraints, including earliest and latest permissible times for event start and end.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "string",
                    "description": "required format: python list of the specific days of the week to filter events for, e.g., ['Monday', 'Friday'] or ['Tuesday']. If none are given use ['Thursday', 'Friday'] as the default return value."
                },
                "min_start_time": {
                    "type": "string",
                    "description": "The earliest permissible start time for events, in 24-hour format, e.g., '08:00' for 8 AM. If none is given use '00:00' as the default return value."
                },
                "max_end_time": {
                    "type": "string",
                    "description": "The latest permissible ending time for events, in 24-hour format, e.g., '17:00' for 5 PM. If noen is given use '23:00' as the default return value."
                },
            }
        }
    }
}]

def get_answer(question, messages, k=112):
    # Convert question to embedding
    question_embedding = get_embedding(question)
    np_question_embedding = np.array(question_embedding)

    # Find k nearest neighbors
    indices, similarities = find_k_nearest_neighbors(
        np_embeddings, np_question_embedding, k
    )

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    chat_response = chat_completion_request(messages, tools)
    assistant_message = chat_response.choices[0].message
    if assistant_message.tool_calls:
        assistant_message.content = str(assistant_message.tool_calls[0].function)
        messages.append({"role": assistant_message.role, "content": assistant_message.content})
        replacements= {"question": question}
        prompt= replace_placeholders(time_template, replacements)
        time_messages= [{"role": "user", "content": prompt}]
        response = chat_completion_request(time_messages, tools=None, model=GPT_MODEL)
        #print(response.choices[0].message.content)
        filters = json.loads(str(response.choices[0].message.content))
        #print(filters)
        filtered_events = filter_events(loaded_documents, filters)
        #print(filtered_events)
        picked_events = []
        counter = 0
        for indice in indices:
            if loaded_documents[indice] in filtered_events:
                picked_events.append(loaded_documents[indice])
                counter+=1
                if counter>= 6:
                    break
        #print(picked_events)
        events_str = ""
        for event in picked_events:
            events_str+= event.content +"\n\n"
        replacements= {"question": question, "events": events_str}
        prompt= replace_placeholders(events_template, replacements)
        #print(prompt)
        events_messages= [{"role": "user", "content": prompt}]
        response = chat_completion_request(events_messages, tools=None, model=GPT_MODEL)
        messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": response.choices[0].message.content})

    else:
        messages.append({"role": assistant_message.role, "content": assistant_message.content})
        return assistant_message.content

    return response.choices[0].message.content

