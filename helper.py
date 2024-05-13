import uuid 
import json
import numpy as np

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
        """ Convert the Document object to a JSON string """
        # Use a dictionary comprehension to handle the renaming of 'id' to 'doc_id'
        data = {k if k != 'id' else 'doc_id': v for k, v in self.__dict__.items()}
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str):
        """ Create a Document object from a JSON string """
        data = json.loads(json_str)
        return cls(**data)

waterkant_festival_description = """The Waterkant Festival is a dynamic and innovative event that takes place annually in Kiel, Germany. Since its inception in 2016, the festival has served as a platform for startups, companies, and creative thinkers to showcase and discuss their ideas on technology, sustainability, and future societal structures. The festival is organized by opencampus.sh, a non-profit association aimed at fostering education and entrepreneurship in Schleswig-Holstein.

Typically held over two days, Waterkant Festival features a diverse array of activities including talks, workshops, and interactive exhibitions. Each year, it gathers hundreds of participants and speakers from various fields—ranging from business and science to politics and the arts—to engage in future-oriented discussions and collaborations. The festival's format includes themed sessions that cover a wide range of topics such as artificial intelligence, sustainability, new work methods, and innovative technology applications.

Waterkant also provides a stage for startups to pitch their ideas and for thought leaders to share insights. The event fosters a strong sense of community and collaboration, with numerous opportunities for networking and co-creation. In addition to professional interactions, the festival incorporates elements of cultural engagement, including live music and social gatherings, creating a vibrant festival atmosphere directly by the waterfront.

Waterkant is notable for its commitment to innovation and its role in promoting the economic and educational development of the region. It supports budding entrepreneurs and established businesses alike, providing a space to explore new ideas and push the boundaries of conventional industry practices​"""

def find_k_nearest_neighbors(matrix, vec, k):
    """Find the k nearest neighbors in the matrix to the vector vec using cosine similarity.
       Returns both the indices of the nearest neighbors and their cosine similarities."""
    # Normalize the input vector and the matrix for cosine similarity
    vec_norm = vec / np.linalg.norm(vec)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    # Compute cosine similarities using matrix multiplication (dot product)
    similarities = np.dot(matrix_norm, vec_norm)

    # Get the indices of the top k similarities using argsort and negative slicing for efficiency
    k_nearest_indices = np.argsort(-similarities)[:k]

    # Get the top k similarities
    k_nearest_distances = similarities[k_nearest_indices]

    return k_nearest_indices, k_nearest_distances

def replace_placeholders(template, replacements):
    for key, value in replacements.items():
        placeholder = "%" + key + "%"
        template = template.replace(placeholder, value)
    return template

