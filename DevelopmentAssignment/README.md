Development Assignment
======================

I've attempted the following tasks:

Level 1 - API Calls (See folder Level_1)

-> I've used gemma2 9 billion as the LLM using Groq. 

-> You can run the level1.py to get output.json as the resultant file.

-> I've set up Groq as mentioned on the website, hence the lack of an API key in the code. Let me know if you need it(it's free!).

Level 2 - Client-Server Model (See folder Level_2_modified)

-> I've created a simple client server model. You can run the server.py and client.py to see the outputs. 

-> Alternatively, you can run the bash script and see the outputs.

-> For the sake of simplicity, I've used a RNG to give names to clients and the outputs are put into json files with the naming scheme: 'response_{client_id}.json'.

-> The server is set up to handle multiple clients at once. But the maximum queries it can handle is 10. Each client can query the server 4 times. You may change this in the init functions of the server and client classes.