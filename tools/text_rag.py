import numpy as np
import heapq
import os
import json

from twelvelabs import TwelveLabs
from langchain.tools import tool

twelvelabs_client = TwelveLabs(api_key=os.getenv('TWELVELABS_API_KEY'))

def _create_text_embed(text: str):

    """

    Creates a text embed using TwelveLabs API and returns the task ID. 
    Args:
        text (str): The text to be embedded.
    
    """

    # TODO: Exercise 2 - Create text embedding using TwelveLabs API
    #
    # Use twelvelabs_client.embed.create() with:
    #   - model_name: 'marengo3.0'
    #   - text: the input text parameter
    #
    # Return: response.text_embedding.segments[0].float_

    pass  # Replace with TwelveLabs API call and return statement

def _load_vector_db():

    """

    Loads the vector database from a JSON file specified by the environment variable VECTOR_DB_FILE.

    Returns:
        dict: A dictionary containing the vector database.
    
    """
    
    try:
        with open(os.getenv('VECTOR_DB_FILE'), 'r') as json_file:
            return json.load(json_file)
    except Exception as e:
        raise ValueError(f"Failed to load vector database: {str(e)}")


@tool("text_rag", description="Performs a text RAG operation using the TwelveLabs API.")
def text_rag(query: str, k: int = 5) -> str:

    """
    Performs a text RAG operation using the TwelveLabs API and returns the top k results.

    Args:
        query (str): The query to search for.
        k (int): The number of results to return.
    Returns:
        str: A string containing the results of the RAG operation.
    
    """

    text_embedding = _create_text_embed(query)
    vector_db = _load_vector_db()

    results = []

    for video_name, video_embedding in vector_db.items():

        video_embedding_array = video_embedding["embedding"]
        start_time, end_time = video_embedding["start_time"], video_embedding["end_time"]
        
        # TODO: Exercise 3 - Calculate cosine similarity
        #
        # Cosine similarity = dot(A, B) / (||A|| * ||B||)
        #
        # Use numpy: np.dot(), np.linalg.norm()
        # Variables: text_embedding, video_embedding_array

        similarity = 0.0  # Replace with cosine similarity calculation

        heapq.heappush(results, (similarity, (video_name, start_time, end_time)))

    top_k_results = heapq.nlargest(k, results)
    result_string = ""
    for result in top_k_results:

        result_string += f"Video: {result[1][0]} | Start Time: {result[1][1]} | End Time: {result[1][2]}\n"
        result_string += f"Similarity: {result[0]}\n"
        result_string += f"-----------------------------------\n"

    return result_string
    
__all__ = ["text_rag"]