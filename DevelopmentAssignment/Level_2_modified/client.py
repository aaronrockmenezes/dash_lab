import socket
import sys
import os
import numpy as np
import json

class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.response_list = []
        self.messages_sent = 0
        self.client_id = np.random.randint(0, 10)
        self.max_queries = 4
    
    def connect(self):
        self.socket.connect((self.host, self.port))

    def send(self, data):
        self.socket.sendall(data)

    def receive(self):
        return self.socket.recv(1024)

    def close(self):
        self.socket.close()

    def read_input_file(self, path):
        with open(path, 'r+') as file:
            line = np.random.choice(file.readlines())
            data = {"query": line.strip()}
            return data
    
    def save_responses(self, path):
        with open(path, 'w+') as file:
            json.dump(self.response_list, file, indent=4)

    def run(self):
        try:
            while True:
                try:
                    if self.messages_sent >= self.max_queries:
                        print(f"Sent {self.max_queries} messages, stopping.")
                        self.close()
                        return 0
                        break
                    data = self.read_input_file('input.txt')
                    print(f"Data to be sent by client {self.client_id}: {data}")
                    self.send(json.dumps(data).encode())
                    self.messages_sent += 1
                    response = self.receive()
                    response = json.loads(response.decode())
                    self.response_list.append(response)
                    self.save_responses(f'responses_{self.client_id}.json')
                except Exception as e:
                    print(f"Error: {e}")
                    print("Emergency saving responses to file.")
                    self.save_responses(f'responses_{self.client_id}.json')
                    self.close()
                    return 0
                    break
            self.save_responses(f'responses_{self.client_id}.json')
        finally:
            self.close()
            return 0

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 2036
    client = Client(host, port)
    client.connect()
    try:
        client.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, stopping.")
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        client.close()
    finally:
        client.close()
