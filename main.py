import os
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()

from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from tools import create_video_embed, text_rag

def _agent_loop(agent: "AgentObject"):

    """

    Loop the prompt to user in CLI and stream the response.

    Args:
        agent (AgentObject): The agent to use for the prompt.

    """

    local_message_history = {
        "messages": [

        ]
    }
    
    while True:

        user_input = input("Enter query: ")
        if user_input.lower() == "exit":
            break

        local_message_history["messages"].append({
            "role": "user",
            "content": user_input
        })

        final_response = ""

        for chunk in agent.stream(local_message_history, stream_mode='updates'):
            for step, data in chunk.items():
                chunk_text = data['messages'][-1].text
                print(chunk_text, end='', flush=True)
                final_response = chunk_text

        local_message_history["messages"].append({
            "role": "assistant",
            "content": final_response
        })

        print("\n")
        
def main():

    if not os.getenv('AWS_ACCESS_KEY_ID'):
        raise ValueError("AWS_ACCESS_KEY_ID not found in environment variables.")
    
    if not os.getenv('AWS_SECRET_ACCESS_KEY'):
        raise ValueError("AWS_SECRET_ACCESS_KEY not found in environment variables.")

    if not os.getenv('TWELVELABS_API_KEY'):
        raise ValueError("TWELVELABS_API_KEY not found in environment variables.")

    if not os.getenv('VECTOR_DB_FILE'):
        raise ValueError("VECTOR_DB_FILE not found in environment variables.")

    # Create an agent using the AWS Bedrock integration and AWS_ACCESS_KEY API key from env.
    model = ChatBedrockConverse(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )

    agent = create_agent(
        model, 
        tools=[create_video_embed, text_rag],
        system_prompt=f"""You are a helpful assistant that creates video embeddings using the TwelveLabs API. The database is a JSON file located at {os.getenv('VECTOR_DB_FILE')}.
            - You can use the create_video_embed tool to create video embeddings.
            - You can use the text_rag tool to perform a text RAG operation.
            - Ensure that you add double slashes within the file path for the create_video_embed tool or any tool that requires a file path to ensure that the file path is valid.
            - Feel free to edit the user's query to make tool calls valid based on the available tools and the context of the query.
        """
    )

    _agent_loop(agent)

if __name__ == "__main__":
    
    main()