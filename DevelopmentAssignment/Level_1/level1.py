import os
import json
from groq import Groq
import time

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Provide relative path to input and output files
input_file_path = "./input.txt"
output_file_name = "output.json"

# Model parameters
MODEL = "gemma2-9b-it"
TEMPERATURE = 0.2
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
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who answers questions in brief, informative responses.",
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

output_dict = {}

# Open input file and read it line by line
with open(input_file_path, "r+") as input_file:
    for line in input_file:
        # Get prompt from input file and passing it through get_chat_completion function
        prompt = line.strip()
        start_time = int(time.time())
        output = get_chat_completion(prompt, MODEL if MODEL is not None else "gemma2-9b-it")
        end_time = int(time.time())

        # Store output in dictionary
        output_dict[prompt] = {
            "Prompt": prompt,
            "Message": output,
            "TimeSent": start_time,
            "TimeRecvd": end_time,
            "Source": MODEL if MODEL is not None else "gemma2-9b-it", 
        }

# Convert dictionary to JSON
output_json = json.dumps(list(output_dict.values()), indent=2)

# Write output to file
with open(output_file_name, "w+") as output_file:
    output_file.write(output_json)
    output_file.close()