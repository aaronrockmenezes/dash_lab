import socket
import signal
import sys
from groq import Groq
import os
import json

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Model parameters
MODEL = "gemma2-9b-it"
TEMPERATURE = 0.1
MAX_TOKENS = 64

# Function to get chat completion
def get_chat_completion(prompt:str, model:str="gemma2-9b-it", temperature:float=0.2, max_tokens:int=64):
    """
    Function to get chat completion from GROQ API

    Parameters
    ----------
    prompt : str
        User input prompt
    model : str, optional
        Model name, by default "gemma2-9b-it"
    temperature : float, optional
        Temperature parameter, by default 0.2
    max_tokens : int, optional
        Maximum tokens to generate, by default 64
    
    Returns
    -------
    str
        Chat completion message
    prompt : str
        User input prompt
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who answers questions in brief, informative responses. Please give replies in one sentence.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return chat_completion.choices[0].message.content, prompt

def signal_handler(sig, frame):
    print("\nServer is shutting down...")
    server_socket.close()
    sys.exit(0)

# Register the signal handler for interrupt and terminate signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
server_socket.bind(('0.0.0.0', 2116))

# Listen for incoming connections
server_socket.listen()

clients = []
num_messages = 0
max_messages = 10

print("Server is running on port 2115...")

try:
    while True:
        # Accept a connection from a client
        client_socket, client_address = server_socket.accept()
        clients.append(client_socket)
        print(f"Connection from {client_address} has been established.")

        # Receive data from the client
        data = client_socket.recv(1024).decode('utf-8')
        if data:
            print(f"Received: {data}")
            
            # Get chat completion from GROQ API
            model_output, prompt = get_chat_completion(data, MODEL, TEMPERATURE, MAX_TOKENS)
            print(f"Response: {model_output}")

            response = {'input_prompt': prompt, 'model_output': model_output, "Source":MODEL}

            # Send the response back to the client
            for client in clients:
                if num_messages >= max_messages:
                    response["End"] = True
                if client != client_socket:
                    response["Source"] = "User"
                    client.send(json.dumps(response).encode('utf-8'))
                else:
                    client.send(json.dumps(response).encode('utf-8'))

        # Close the client connection
        if num_messages >= max_messages:
            for client_socket in clients:
                client_socket.close()
            break

except Exception as e:
    print(f"An error occurred: {e}")
    server_socket.close()

finally:
    print("Cleaning up the server socket...")
    server_socket.close()
