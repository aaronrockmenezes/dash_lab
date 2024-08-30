#!/bin/bash

# Start server in the background
python3 server.py &

# Start 3 clients in the background
for i in {1..1}
do
  python3 client.py &
done

# Wait for all background processes to complete
wait
