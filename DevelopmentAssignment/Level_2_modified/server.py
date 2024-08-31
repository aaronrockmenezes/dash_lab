import socket
import sys
import os
from groq import Groq
import time
import threading
import json
import copy

class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_list = []
        self.client_addresses = []
        self.max_queries = 10
        self.num_queries = 0
        self.groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])
        self.model = "gemma2-9b-it"


    def run_llm(self, prompt:str, model:str="gemma2-9b-it", temperature:float=0.2, max_tokens:int=64):
        chat_completion = self.groq_client.chat.completions.create(
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
        return chat_completion.choices[0].message.content

    def assemble_response(self, data):
        start_time = int(time.time())
        output = self.run_llm(data['query'], self.model)
        end_time = int(time.time())
        response = {
            "Prompt": data['query'],
            "Response": output,
            "TimeSent": start_time,
            "TimeReceived": end_time,
            "Source": self.model
        }
        # print(f"Response: {response}")
        return response

    def handle_client(self, client_socket):
        while True:
            data = client_socket.recv(1024)
            if data:
                # print(f"Received data1: {data}")
                if self.num_queries >= self.max_queries:
                    print(f"Reached maximum queries, stopping.")
                    self.close()
                    return 0
                    break
                data = json.loads(data.decode())
                if not data:
                    print(f"Received empty data - {data}")
                    print(f"Closing connection")
                    self.close()
                    break
                self.num_queries += 1
                response = self.assemble_response(data)
                self.broadcast(response, sender=client_socket)
        print(f"Closing connection for client - {client_socket}")
        client_socket.close()

    def start(self):
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        print(f"Server listening on {self.host}:{self.port}")
        try:
            while True:
                if self.num_queries >= self.max_queries:
                    print(f"Reached maximum queries, stopping.")
                    self.close()
                    break
                client_socket, addr = self.sock.accept()
                print(f"New connection from {addr[0]}:{addr[1]}")
                self.client_list.append(client_socket)
                self.client_addresses.append(addr)
                print(f"Connected clients: {[i for i in self.client_addresses]}")
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_thread.start()
        except Exception as e:
            print(f"Error: {e}")
            print("Closing server")
            self.close()
        finally:
            print("Closing server")
            self.close()

    def close(self):
        self.sock.close()
    
    def broadcast(self, message, sender):
        for client in self.client_list:
            if client != sender:
                reply = message.copy()
                reply.update({'Source':'user'})
                reply = json.dumps(reply)
                client.sendall(reply.encode())
            else:
                reply = message.copy()
                reply = json.dumps(reply)
                client.sendall(reply.encode())


if __name__ == "__main__":
    host = '127.0.0.1'
    port = 2036
    server = Server(host, port)
    try:
        server.start()
    except KeyboardInterrupt:
        print("Closing server")
        server.close()
    finally:
        print("Closing server")
        server.close()