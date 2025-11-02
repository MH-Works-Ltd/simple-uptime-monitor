#!/usr/bin/env bash
set -e

REPO_URL="https://raw.githubusercontent.com/MH-Works-Ltd/simple-uptime-monitor/main"

# Create working directory
mkdir -p simple-uptime-monitor
cd simple-uptime-monitor

# Download files
echo "Downloading files..."
curl -sO "$REPO_URL/run.sh"
curl -sO "$REPO_URL/main.py"
curl -sO "$REPO_URL/urls.txt.example"

# Make script executable
chmod +x run.sh

# Rename example file to urls.txt if it doesn't exist
if [ ! -f urls.txt ]; then
  cp urls.txt.example urls.txt
fi

echo "✅ Setup complete!"
echo "Run the monitor using:"
echo "  ./run.sh"
