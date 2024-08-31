import numpy as np
import json
import os

data = [
    {"name": "John", "age": 30, "city": "New York"},
    {"name": "Alice", "age": 25, "city": "Los Angeles"},
    {"name": "Bob", "age": 35, "city": "Chicago"}
]

with open('/home/aaron/Desktop/dash_lab/nope/op.json', 'w') as f:
    json.dump(data, f, indent=4)