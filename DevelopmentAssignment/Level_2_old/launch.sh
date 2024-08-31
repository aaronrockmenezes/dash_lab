#!/bin/bash

# Start server in the background
python3 server.py &

sleep 2

# Start 3 clients in the background
for i in {1..5}
do
  python3 client.py &
done

# Wait for all background processes to complete
wait
