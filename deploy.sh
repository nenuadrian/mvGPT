#!/usr/bin/env bash
# Builds the app on a server and runs it in Docker, on port 7860: ./deploy.sh debian@203.0.113.7
set -euo pipefail
server=$1

# The committed files of HEAD, as a tar stream, unpacked into ~/mvgpt on the server.
git archive HEAD | ssh "$server" 'rm -rf mvgpt && mkdir mvgpt && tar -x -C mvgpt'
# Build the image there, and swap the running container for one from it. The volume keeps the
# models downloaded from the Hub across deploys.
ssh "$server" 'cd mvgpt && docker build -q -t mvgpt . && docker rm -f mvgpt > /dev/null 2>&1;
  docker run -d --name mvgpt --restart unless-stopped -p 7860:7860 -v mvgpt-hub:/root/.cache/huggingface mvgpt'
