import json
import uuid
import numpy as np
from openai import OpenAI
import streamlit as st


waterkant_festival_description = """The Waterkant Festival is a dynamic and innovative event that takes place annually in Kiel, Germany. Since its inception in 2016, the festival has served as a platform for startups, companies, and creative thinkers to showcase and discuss their ideas on technology, sustainability, and future societal structures. The festival is organized by opencampus.sh, a non-profit association aimed at fostering education and entrepreneurship in Schleswig-Holstein.

Typically held over two days, Waterkant Festival features a diverse array of activities including talks, workshops, and interactive exhibitions. Each year, it gathers hundreds of participants and speakers from various fields—ranging from business and science to politics and the arts—to engage in future-oriented discussions and collaborations. The festival's format includes themed sessions that cover a wide range of topics such as artificial intelligence, sustainability, new work methods, and innovative technology applications.

Waterkant also provides a stage for startups to pitch their ideas and for thought leaders to share insights. The event fosters a strong sense of community and collaboration, with numerous opportunities for networking and co-creation. In addition to professional interactions, the festival incorporates elements of cultural engagement, including live music and social gatherings, creating a vibrant festival atmosphere directly by the waterfront.

Waterkant is notable for its commitment to innovation and its role in promoting the economic and educational development of the region. It supports budding entrepreneurs and established businesses alike, providing a space to explore new ideas and push the boundaries of conventional industry practices​"""

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


class Document:
    def __init__(self, content, metadata=None, doc_id=None, num_chunks=None):
        self.content = content
        self.metadata = metadata if metadata is not None else {}
        self.id = doc_id if doc_id is not None else str(uuid.uuid4())
        self.num_chunks = num_chunks

    def add_metadata(self, key, value):
        """
        Adds a key-value pair to the document's metadata.

        :param key: The key for the metadata item.
        :param value: The value for the metadata item.
        """
        self.metadata[key] = value

    def __str__(self):
        """
        Returns a string representation of the document.

        :return: A string containing the content and metadata of the document.
        """
        return f"Content: {self.content}\nMetadata: {self.metadata}"

    def to_json(self):
        """Convert the Document object to a JSON string"""
        # Use a dictionary comprehension to handle the renaming of 'id' to 'doc_id'
        data = {k if k != "id" else "doc_id": v for k, v in self.__dict__.items()}
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str):
        """Create a Document object from a JSON string"""
        data = json.loads(json_str)
        return cls(**data)


np_embeddings = np.load("prep/embeddings.npy")


def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product


def find_k_nearest_neighbors(matrix, vec, k):
    """Find the k nearest neighbors in the matrix to the vector vec using cosine similarity.
    Returns both the indices of the nearest neighbors and their cosine similarities."""
    # Calculate cosine similarity with each vector in the matrix
    similarities = np.array(
        [cosine_similarity(vec, matrix[i]) for i in range(len(matrix))]
    )

    # Get the indices of the top k similarities
    k_nearest_indices = np.argsort(-similarities)[:k]

    # Get the top k similarities
    k_nearest_distances = similarities[k_nearest_indices]

    return k_nearest_indices, k_nearest_distances


with open("prep/documents.json", "r") as file:
    json_documents = json.load(file)

loaded_documents = [Document.from_json(json_str) for json_str in json_documents]

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def replace_placeholders(template, replacements):
    for key, value in replacements.items():
        placeholder = "%" + key + "%"
        template = template.replace(placeholder, value)
    return template


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
