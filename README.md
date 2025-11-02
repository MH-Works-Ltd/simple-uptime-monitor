# Simple Uptime Monitor

A lightweight, Tailnet-friendly health monitor with ntfy alerts and a Rich TUI interface. Monitor your services and get notified when they go down or recover.

## Features

- **Real-time monitoring** of HTTP endpoints with configurable intervals
- **ntfy notifications** for downtime alerts and recovery notifications
- **Rich terminal UI** showing live status, latency, and statistics
- **Exponential backoff** for repeated alerts to avoid notification spam
- **Heartbeat notifications** to confirm the monitor is still running
- **Tailscale-friendly** for monitoring internal services
- **Concurrent checks** for efficient monitoring of multiple endpoints

## Installation

1. Download and run the installation script:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/MH-Works-Ltd/simple-uptime-monitor/main/install.sh | bash
   ```

2. Configure your URLs by editing `urls.txt` (see `urls.txt.example` for the format):
   ```bash
   # ntfy: ntfy.sh/your_channel_name
   # One URL per line; comments and blanks are ignored
   https://internal.mycompany.com/api/health
   https://external.service/api/health
   ```

3. Run the monitor (typically in a screen or tmux session):
   ```bash
   ./run.sh
   ```

## Usage

The monitor will:
- Display a live table showing the status of all configured URLs
- Send ntfy notifications when services go down or recover
- Send periodic heartbeat notifications to confirm it's still running

### Command Line Options

- `--urls`: Path to URLs file (required)
- `--interval-seconds`: Ping interval per URL (default: 60)
- `--timeout-seconds`: HTTP timeout per request (default: 10)
- `--max-backoff-minutes`: Cap for exponential re-alerts (default: 30)
- `--resp-bytes`: Max response bytes in notifications (default: 200)
- `--heartbeat-hours`: Heartbeat interval (default: 24.0)
- `--insecure`: Skip TLS verification
- `--force-terminal`: Force terminal mode (useful for SSH sessions where terminal detection may fail)

## Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager
- httpx and rich libraries (automatically installed via uv)
