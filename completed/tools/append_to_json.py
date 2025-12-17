import json

from pathlib import Path
from langchain.tools import tool

@tool("append_to_json", description="Appends a dictionary as a new entry to a JSON file.")
def append_to_json(file_path: str, content: dict) -> str:

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
        return f"File not found at path: {file_path}"
    
    data = {}

    try:

        with open(file_path, 'r') as json_file:
            raw_json_content = json_file.read().strip()
            if raw_json_content:
                data = json.loads(raw_json_content)

        print(f"Read existing JSON content from {file_path}. Data keys: {list(data.keys())}")
        
        with open(file_path, 'w', encoding='utf-8') as json_file:
            data.update(content)
            json_file.seek(0)
            json.dump(data, json_file, indent=4)

        return f"Content appended to {file_path} successfully."
    
    except Exception as e:

        return f"An error occurred: {str(e)}"

__all__ = ["append_to_json"]