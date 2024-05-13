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

smileys = [
    {"emoji": "🌊", "description": "Wave emoji for water and sea themes."},
    {
        "emoji": "⛵",
        "description": "Sailboat to represent sailing or boat-related activities.",
    },
    {
        "emoji": "🚤",
        "description": "Speedboat for more energetic or fast-paced water activities.",
    },
    {"emoji": "🐟", "description": "Fish emoji to symbolize marine life."},
    {"emoji": "🎣", "description": "Fishing pole for fishing activities or themes."},
    {"emoji": "🏄", "description": "Surfer for surfing or beach-related fun."},
    {
        "emoji": "🌅",
        "description": "Sunrise/sunset over water for beautiful waterfront scenes.",
    },
    {
        "emoji": "🍹",
        "description": "Tropical drink to represent relaxing by the water.",
    },
    {
        "emoji": "🎶",
        "description": "Musical notes for live music or performances at the festival.",
    },
    {"emoji": "🎆", "description": "Fireworks for evening celebrations by the water."},
]

np_embeddings = np.load("prep/embeddings.npy")

with open("prep/documents.json", "r") as file:
    json_documents = json.load(file)

loaded_documents = [Document.from_json(json_str) for json_str in json_documents]

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

template = (
    "General info\n\n"
    + waterkant_festival_description
    + "\n--------------\n"
    + "Here is a list of events which could be helpful to answer the question:\n\n%events%Answer the question of the Waterkant Festival based on the information above and in the language of the asked question!. Here is the question:\n%question%"
)

system_prompt = "You are a helpful chatbot for the attendees of the Waterkant Festival! You answer questions about events, schedules, venue information, transportation options, or anything else. The attendees should feel free to ask you. You respond in the same language the user asks the question!"


def get_prompt(question, k=10):
    # Convert question to embedding
    question_embedding = get_embedding(question)
    np_question_embedding = np.array(question_embedding)

    # Find k nearest neighbors
    indices, similarities = find_k_nearest_neighbors(
        np_embeddings, np_question_embedding, k
    )

    # Retrieve the documents corresponding to the indices
    events = [loaded_documents[index] for index in indices]

    # Concatenate the content of each document
    events_str = ""
    for event in events:
        events_str += event.content + "\n\n"

    # Prepare replacements for placeholders in the template
    replacements = {"question": question, "events": events_str}

    # Replace placeholders in the template to create the final prompt
    prompt = replace_placeholders(template, replacements)

    return prompt