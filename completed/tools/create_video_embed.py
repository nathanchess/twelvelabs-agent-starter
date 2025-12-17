import os
import base64
import json

from pathlib import Path
from twelvelabs import TwelveLabs, VideoInputRequest, MediaSource
from langchain.tools import tool

twelvelabs_client = TwelveLabs(api_key=os.getenv('TWELVELABS_API_KEY'))

def _append_to_json(file_path: str, content: dict) -> str:

    """
    Appends a dictionary as a new entry to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        content (dict): The dictionary content to append.

    Returns:
        str: A confirmation message.
    
    """

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found at path: {file_path}")
    
    data = {}

    with open(file_path, 'r') as json_file:
        raw_json_content = json_file.read().strip()
        if raw_json_content:
            data = json.loads(raw_json_content)
    
    with open(file_path, 'w', encoding='utf-8') as json_file:
        data.update(content)
        json_file.seek(0)
        json.dump(data, json_file, indent=4)

    return f"Content appended to {file_path} successfully."

@tool("create_video_embed", description="Creates a video embed using TwelveLabs API and returns the embedding.")
def create_video_embed(video_file_path: str) -> str:

    """
    
    Creates a video embed using TwelveLabs API and returns embedding. 

    Args:
        video_file_path (str): The file path to the video to be uploaded.
    
    """

    video_path = Path(video_file_path)
    if not video_path.exists():
        return f"Video file not found at path: {video_file_path}"

    data = {}

    with open(os.getenv('VECTOR_DB_FILE'), 'r') as json_file:
        raw_json_content = json_file.read().strip()
        if raw_json_content:
            data = json.loads(raw_json_content)

    video_name = video_path.name

    if video_name in data:
        return f"Video {video_name} already exists in the database."
    
    with open(video_file_path, "rb") as video_file:
        video_bytes = video_file.read()
        base64_video_string = base64.b64encode(video_bytes).decode('utf-8')

    print(f"File at {video_file_path} read and encoded to base64.")
    
    response = twelvelabs_client.embed.v_2.create(
        input_type='video',
        model_name='marengo3.0',
        video=VideoInputRequest(
            media_source=MediaSource(
                base_64_string = base64_video_string
            ),
        ),
    )
    
    print(f"Created embed task with Marengo 3.0 | Embedding Size: {len(response.data[0].embedding)}")
    print(f'Number of embeddings: {len(response.data)}')

    for i in range(len(response.data)):
        embedding_data = response.data[i]

        start_time, end_time = embedding_data.start_sec, embedding_data.end_sec
        embedding = embedding_data.embedding
        embedding_name = f"{video_name}_{start_time:.2f}_{end_time:.2f}"
        
        _append_to_json(os.getenv('VECTOR_DB_FILE'), {
            embedding_name: {
                "embedding": embedding,
                "start_time": start_time,
                "end_time": end_time
            },
        })

    return f"Embedding appended to {os.getenv('VECTOR_DB_FILE')} successfully. Model: Marengo 3.0 | Number of embeddings: {len(response.data)} | Video Name: {video_name} | Embedding Size: {len(response.data[0].embedding)}"

__all__ = ["create_video_embed"]