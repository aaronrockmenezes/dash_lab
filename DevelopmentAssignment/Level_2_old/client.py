import socket
import numpy as np
import json
import os
import time

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the server address and port
server_address = ('localhost', 2116)

print("Client is connected to server running on port 2115...")

try:
    # Connect to the server
    client_socket.connect(server_address)

    # Generate a random client ID
    client_id = np.random.randint(1000, 9999)

    with open('./input.txt', 'r+') as file:
        line = np.random.choice(file.readlines())
        message = line.strip()

    # Send the message to the server
    client_socket.sendall(message.encode('utf-8'))
    start_time = int(time.time())

    # Receive the server response
    response = client_socket.recv(1024).decode('utf-8')

    response = json.loads(response)
    prompt = response['input_prompt']
    model_output = response['model_output']

    print(f"Received: {model_output}")
    end_time = int(time.time())

    # Append response to output json file (first check if file exists)
    output_dict = {}
    output_file_name = f"output_{client_id}.json"
    
    output_dict[prompt] = {
        "Prompt": prompt,
        "Message": model_output,
        "TimeSent": start_time,
        "TimeRecvd": end_time,
        "ClientID": client_id,
        "Source": response["Source"]
    }

    with open(output_file_name, "w+") as output_file:
        json.dump(list(output_dict.values())[0], output_file, indent=4)

    # Close the connection
    if response["End"] and response["End"]==True:
        client_socket.close()

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Cleaning up the client socket...")
    client_socket.close()