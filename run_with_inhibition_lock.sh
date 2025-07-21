#!/bin/bash

# Prevent system sleep
sudo systemd-inhibit --what=handle-lid-switch:sleep --why="Running simulate_events" sleep infinity &
INHIBIT_PID=$!

# Run Docker job
docker-compose run ai-recommendation-system python cli.py simulate_events

# Cleanup lock after job
sudo kill $INHIBIT_PID
