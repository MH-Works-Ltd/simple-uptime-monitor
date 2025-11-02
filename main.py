#!/usr/bin/env python3
import argparse
import asyncio
import datetime as dt
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import time

import httpx
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich import box

# ------------------
# Config & CLI
# ------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tailnet-friendly health monitor with ntfy alerts & Rich TUI.")
    p.add_argument("--urls", required=True, help="Path to a .txt file with one URL per line. First line should be '# ntfy: <topic>' to specify ntfy channel.")
    p.add_argument("--interval-seconds", type=int, default=60, help="Ping interval per URL.")
    p.add_argument("--timeout-seconds", type=float, default=10, help="HTTP timeout per request.")
    p.add_argument("--max-backoff-minutes", type=int, default=30, help="Cap for exponential re-alerts while down.")
    p.add_argument("--resp-bytes", type=int, default=200, help="Max response bytes to include in notifications.")
    p.add_argument("--heartbeat-hours", type=float, default=24.0, help="Send an 'I am alive' heartbeat this often.")
    p.add_argument("--insecure", action="store_true", help="Skip TLS verification.")
    p.add_argument("--force-terminal", action="store_true", help="Force terminal mode for SSH sessions.")
    return p.parse_args()

def load_urls(path: str) -> tuple[str, List[str]]:
    """Load ntfy topic and URLs from file. Returns (ntfy_topic, urls)."""
    ntfy_topic = None
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Check for ntfy config line
            if line.startswith("# ntfy:"):
                ntfy_topic = line.split(":", 1)[1].strip()
                continue
            # Skip other comments
            if line.startswith("#"):
                continue
            urls.append(line)
    if not ntfy_topic:
        raise SystemExit("No ntfy topic found. First line should be '# ntfy: <topic>' (e.g., '# ntfy: ntfy.sh/my_alerts')")
    if not urls:
        raise SystemExit("No URLs found in the file.")
    return ntfy_topic, urls

def normalize_ntfy(topic_or_url: str) -> str:
    if topic_or_url.startswith("http://") or topic_or_url.startswith("https://"):
        return topic_or_url.rstrip("/")
    if topic_or_url.startswith("ntfy.sh/"):
        return "https://" + topic_or_url.rstrip("/")
    # Plain topic
    return f"https://ntfy.sh/{topic_or_url.strip('/')}"

# ------------------
# State
# ------------------

@dataclass
class TargetState:
    url: str
    last_status: str = "pending"            # "good" | "bad" | "pending"
    last_http_code: Optional[int] = None
    last_latency_ms: Optional[int] = None
    last_error: Optional[str] = None
    last_response_snippet: Optional[str] = None
    last_change_ts: Optional[dt.datetime] = None
    last_check_ts: Optional[dt.datetime] = None
    next_check_ts: Optional[dt.datetime] = None

    # Alert/backoff
    is_bad: bool = False
    backoff_minutes: int = 0
    next_notify_ts: Optional[dt.datetime] = None

    # Stats
    checks_ok: int = 0
    checks_fail: int = 0
    
    # Historical latency data: (timestamp, latency_ms)
    latency_history: deque = field(default_factory=lambda: deque(maxlen=1440))  # 24h at 1min intervals

# ------------------
# Notifier
# ------------------

class Notifier:
    def __init__(self, ntfy_url: str):
        self.ntfy_url = ntfy_url
        self.console = Console()

    async def send(self, title: str, message: str, priority: str = "default"):
        headers = {
            "Title": title,
            "Priority": priority,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                await client.post(self.ntfy_url, content=message.encode("utf-8", errors="replace"), headers=headers)
            except Exception as e:
                # We log to console but don't raise—notifications shouldn't kill the monitor.
                self.console.print(f"[yellow]Warning:[/yellow] failed to send ntfy notification: {e}")

# ------------------
# Monitor
# ------------------

class Monitor:
    def __init__(
        self,
        urls: List[str],
        interval_seconds: int,
        timeout_seconds: float,
        max_backoff_minutes: int,
        resp_bytes: int,
        insecure: bool,
        notifier: Notifier,
        force_terminal: bool = False,
    ):
        self.urls = urls
        self.interval = interval_seconds
        self.timeout = timeout_seconds
        self.max_backoff = max_backoff_minutes
        self.resp_bytes = resp_bytes
        self.verify = not insecure
        self.notifier = notifier
        self.console = Console(force_terminal=force_terminal)

        self.targets: Dict[str, TargetState] = {u: TargetState(url=u) for u in urls}
        self.start_ts = dt.datetime.now()
        self._stop = asyncio.Event()

    async def ping_once(self, url: str):
        now = dt.datetime.now()
        st = self.targets[url]
        st.last_check_ts = now

        t0 = time.perf_counter()
        code = None
        err = None
        body_snippet = None

        try:
            async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify) as client:
                resp = await client.get(url)
                code = resp.status_code
                content_bytes = resp.content[: self.resp_bytes]
                try:
                    body_snippet = content_bytes.decode("utf-8", errors="replace")
                except Exception:
                    body_snippet = repr(content_bytes)
        except Exception as e:
            err = str(e)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        good = (code is not None) and (200 <= code < 300) and (err is None)

        st.last_http_code = code
        st.last_latency_ms = latency_ms
        st.last_error = err
        st.last_response_snippet = body_snippet
        
        # Store latency in history (with timestamp)
        if latency_ms is not None:
            st.latency_history.append((now, latency_ms))

        previous_status = st.last_status
        st.last_status = "good" if good else "bad"
        st.is_bad = not good

        if good:
            st.checks_ok += 1
            # if we recovered from bad -> notify recovery & reset backoff
            if previous_status == "bad":
                await self.notifier.send(
                    title="RECOVERED",
                    message=self._format_message(url, code, latency_ms, body_snippet, recovered=True),
                    priority="default",
                )
                st.backoff_minutes = 0
                st.next_notify_ts = None
                st.last_change_ts = now
            elif previous_status != "good":
                st.last_change_ts = now
        else:
            st.checks_fail += 1

            # just transitioned to bad -> immediate alert
            if previous_status != "bad":
                st.backoff_minutes = 1
                st.next_notify_ts = now  # fire immediately
                st.last_change_ts = now

            # Decide whether to send a re-alert
            if st.next_notify_ts is None or now >= st.next_notify_ts:
                # send
                await self.notifier.send(
                    title="DOWN",
                    message=self._format_message(url, code, latency_ms, body_snippet, error=err),
                    priority="high",
                )
                # schedule next re-alert with exponential backoff
                st.backoff_minutes = min(st.backoff_minutes * 2 if st.backoff_minutes else 1, self.max_backoff)
                # If just transitioned to bad, we already set 1; if this is a re-alert, it doubles.
                if previous_status == "bad":
                    # ensure we doubled at least once
                    if st.backoff_minutes == 0:
                        st.backoff_minutes = 1
                st.next_notify_ts = now + dt.timedelta(minutes=st.backoff_minutes)

    def _format_message(self, url, code, latency_ms, snippet, recovered=False, error: Optional[str] = None):
        parts = [f"URL: {url}"]
        if recovered:
            parts.append("Status: RECOVERED")
        elif error:
            parts.append(f"Status: DOWN (exception)")
            parts.append(f"Error: {error}")
        else:
            parts.append(f"Status: DOWN (HTTP {code})")

        parts.append(f"Latency: {latency_ms} ms")
        if snippet:
            # Keep message short-ish; snippet is already truncated at fetch time.
            parts.append("--- Response (truncated) ---")
            parts.append(snippet)
        return "\n".join(parts)

    def _render_chart(self) -> Panel:
        """Render a text-based line chart of latency over time for all targets."""
        CHART_WIDTH = 100
        CHART_HEIGHT = 20
        
        # Colors for different services (cycling through rich color names)
        colors = ["cyan", "magenta", "green", "yellow", "blue", "red", "bright_cyan", "bright_magenta"]
        
        # Collect all data points from last 24 hours
        now = dt.datetime.now()
        cutoff_time = now - dt.timedelta(hours=24)
        
        # Filter and organize data
        service_data: Dict[str, List[Tuple[dt.datetime, int]]] = {}
        all_latencies = []
        
        for url, st in self.targets.items():
            # Filter to last 24 hours
            recent_data = [(ts, lat) for ts, lat in st.latency_history if ts >= cutoff_time]
            if recent_data:
                service_data[url] = recent_data
                all_latencies.extend([lat for _, lat in recent_data])
        
        if not all_latencies:
            return Panel("[dim]No latency data available yet[/dim]", title="Latency Over Time (Last 24h)", border_style="blue")
        
        # Calculate scale
        min_lat = min(all_latencies)
        max_lat = max(all_latencies)
        lat_range = max_lat - min_lat if max_lat > min_lat else 1
        
        # Time range
        if service_data:
            all_times = [ts for data in service_data.values() for ts, _ in data]
            min_time = min(all_times)
            max_time = max(all_times)
            time_range = (max_time - min_time).total_seconds()
            if time_range == 0:
                time_range = 1
        else:
            min_time = now - dt.timedelta(hours=24)
            max_time = now
            time_range = 86400
        
        # Create canvas
        canvas = [[' ' for _ in range(CHART_WIDTH)] for _ in range(CHART_HEIGHT)]
        color_map = [[None for _ in range(CHART_WIDTH)] for _ in range(CHART_HEIGHT)]
        
        # Draw grid lines (horizontal)
        for i in range(0, CHART_HEIGHT, 5):
            for j in range(CHART_WIDTH):
                if canvas[i][j] == ' ':
                    canvas[i][j] = '·'
        
        # Plot each service
        for idx, (url, data) in enumerate(service_data.items()):
            color = colors[idx % len(colors)]
            
            # Convert data points to canvas coordinates
            points = []
            for ts, lat in data:
                # X: time position (0 to CHART_WIDTH-1)
                time_offset = (ts - min_time).total_seconds()
                x = int((time_offset / time_range) * (CHART_WIDTH - 1))
                x = max(0, min(CHART_WIDTH - 1, x))
                
                # Y: latency position (inverted: 0 at top is max, bottom is min)
                y = int(((max_lat - lat) / lat_range) * (CHART_HEIGHT - 1))
                y = max(0, min(CHART_HEIGHT - 1, y))
                
                points.append((x, y))
            
            # Draw lines between consecutive points
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                
                # Simple line drawing (Bresenham-ish)
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                err = dx - dy
                
                x, y = x1, y1
                while True:
                    # Choose character based on direction
                    if x == x1 and y == y1:
                        char = '●'
                    elif dx > dy:
                        char = '─'
                    else:
                        char = '│'
                    
                    canvas[y][x] = char
                    color_map[y][x] = color
                    
                    if x == x2 and y == y2:
                        break
                    
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x += sx
                    if e2 < dx:
                        err += dx
                        y += sy
        
        # Build the chart text with colors
        chart_lines = []
        
        # Y-axis labels and chart
        y_labels_width = 8
        for i in range(CHART_HEIGHT):
            # Y-axis label (latency value)
            lat_value = max_lat - (i / (CHART_HEIGHT - 1)) * lat_range
            y_label = f"{int(lat_value):>6}ms"
            
            # Build colored line
            line = Text(y_label + " ┤ ")
            for j in range(CHART_WIDTH):
                char = canvas[i][j]
                color = color_map[i][j]
                if color:
                    line.append(char, style=color)
                else:
                    line.append(char, style="dim")
            
            chart_lines.append(line)
        
        # X-axis
        x_axis = Text(" " * y_labels_width + " └" + "─" * CHART_WIDTH)
        chart_lines.append(x_axis)
        
        # X-axis labels (time)
        time_labels = Text(" " * (y_labels_width + 2))
        # Show times at start, middle, and end
        positions = [(0, 0.0), (CHART_WIDTH // 2, 0.5), (CHART_WIDTH - 10, 1.0)]
        for pos_idx, (pos, fraction) in enumerate(positions):
            time_val = min_time + dt.timedelta(seconds=time_range * fraction)
            time_str = time_val.strftime("%H:%M")
            if pos_idx > 0:
                # Add spacing
                time_labels.append(" " * (pos - len(time_labels.plain)))
            time_labels.append(time_str)
        
        chart_lines.append(time_labels)
        chart_lines.append(Text(" " * (y_labels_width + CHART_WIDTH // 2 - 5) + "Time (24h)", style="dim"))
        
        # Legend
        legend = Text("\n\nLegend: ", style="bold")
        for idx, url in enumerate(service_data.keys()):
            if idx > 0:
                legend.append(" • ")
            color = colors[idx % len(colors)]
            # Shorten URL for legend
            short_url = url if len(url) <= 50 else url[:47] + "..."
            legend.append("■", style=color)
            legend.append(f" {short_url}", style=color)
            if (idx + 1) % 2 == 0 and idx < len(service_data) - 1:
                legend.append("\n        ")
        
        chart_lines.append(legend)
        
        # Combine all lines
        final_text = Text()
        for line in chart_lines:
            final_text.append(line)
            final_text.append("\n")
        
        return Panel(
            final_text,
            title=f"[bold]Latency Over Time (Last 24h)[/bold]  •  Range: {min_lat}ms - {max_lat}ms",
            border_style="blue",
            padding=(1, 2),
        )

    async def run(self, heartbeat_hours: float):
        # Send a startup heartbeat
        url_list = "\n".join(f"• {url}" for url in self.urls)
        await self.notifier.send(
            title="Monitor started",
            message=f"Monitoring {len(self.urls)} URL(s). Heartbeat every {heartbeat_hours}h.\n\nURLs:\n{url_list}",
            priority="default",
        )

        # Perform initial ping cycle before displaying table (no console output - Live display will show status)
        await asyncio.gather(*(self.ping_once(u) for u in self.urls))

        # next heartbeat time
        next_heartbeat = dt.datetime.now() + dt.timedelta(hours=heartbeat_hours)

        # Set initial next_check_ts for all targets
        initial_next_check = dt.datetime.now() + dt.timedelta(seconds=self.interval)
        for st in self.targets.values():
            st.next_check_ts = initial_next_check

        # schedule ticks: we check all urls every interval
        # Create layout with table on top and chart on bottom
        layout = Layout()
        layout.split_column(
            Layout(name="table", ratio=1),
            Layout(name="chart", ratio=2)
        )
        
        with Live(layout, console=self.console, refresh_per_second=4, auto_refresh=False) as live:
            while not self._stop.is_set():
                tick_started = dt.datetime.now()

                # Ping all targets concurrently
                await asyncio.gather(*(self.ping_once(u) for u in self.urls))

                # Update next check time for all targets
                next_check = dt.datetime.now() + dt.timedelta(seconds=self.interval)
                for st in self.targets.values():
                    st.next_check_ts = next_check

                # Heartbeat?
                now = dt.datetime.now()
                if now >= next_heartbeat:
                    ok = sum(1 for t in self.targets.values() if t.last_status == "good")
                    bad = sum(1 for t in self.targets.values() if t.last_status == "bad")
                    url_list = "\n".join(f"• {url}" for url in self.urls)
                    await self.notifier.send(
                        title="Heartbeat",
                        message=f"Still running. Targets good: {ok}, bad: {bad}. Next heartbeat in {heartbeat_hours}h.\n\nURLs:\n{url_list}",
                        priority="default",
                    )
                    next_heartbeat = now + dt.timedelta(hours=heartbeat_hours)

                # Update display with latest data after ping cycle completes
                layout["table"].update(self._render_table())
                layout["chart"].update(self._render_chart())
                live.refresh()  # Manually refresh only when we have new data
                
                # Sleep until next interval (keeping us close to a fixed cadence)
                elapsed = (dt.datetime.now() - tick_started).total_seconds()
                sleep_for = max(0.0, self.interval - elapsed)
                # Just sleep without refreshing display (no flicker on SSH)
                end_sleep = time.perf_counter() + sleep_for
                while time.perf_counter() < end_sleep and not self._stop.is_set():
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=0.25)
                    except asyncio.TimeoutError:
                        pass

    def stop(self):
        self._stop.set()

    def _render_table(self) -> Table:
        table = Table(
            title=f"Private Health Monitor • {len(self.urls)} target(s) • started {self.start_ts.strftime('%Y-%m-%d %H:%M:%S')}",
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        table.add_column("URL", overflow="fold")
        table.add_column("Status", justify="center")
        table.add_column("Code", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Last Check", justify="right")
        table.add_column("Since", justify="right")
        table.add_column("Next Alert", justify="right")
        table.add_column("OK/Fail", justify="right")

        for u, st in self.targets.items():
            status = st.last_status
            status_str = {
                "good": "[green]GOOD[/green]",
                "bad": "[red]BAD[/red]",
                "pending": "[blue]PENDING[/blue]",
            }.get(status, status or "pending")

            code_str = str(st.last_http_code) if st.last_http_code is not None else "-"
            lat_str = f"{st.last_latency_ms} ms" if st.last_latency_ms is not None else "-"

            if st.last_check_ts:
                last_check_str = st.last_check_ts.strftime("%H:%M:%S")
            else:
                last_check_str = "-"

            if st.last_change_ts:
                since_str = st.last_change_ts.strftime("%H:%M:%S")
            else:
                since_str = "-"

            if st.is_bad and st.next_notify_ts:
                next_alert_str = st.next_notify_ts.strftime("%H:%M:%S")
            else:
                next_alert_str = "-"

            okfail = f"{st.checks_ok}/{st.checks_fail}"

            table.add_row(u, status_str, code_str, lat_str, last_check_str, since_str, next_alert_str, okfail)

        return table

# ------------------
# Utilities
# ------------------

def human_delta(td: dt.timedelta) -> str:
    total = int(abs(td.total_seconds()))
    sign = "-" if td.total_seconds() < 0 else ""
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{sign}{days}d {hours}h"
    if hours:
        return f"{sign}{hours}h {minutes}m"
    if minutes:
        return f"{sign}{minutes}m {seconds}s"
    return f"{sign}{seconds}s"

# ------------------
# Main
# ------------------

async def amain():
    args = parse_args()
    ntfy_topic, urls = load_urls(args.urls)
    ntfy_url = normalize_ntfy(ntfy_topic)

    notifier = Notifier(ntfy_url)
    monitor = Monitor(
        urls=urls,
        interval_seconds=args.interval_seconds,
        timeout_seconds=args.timeout_seconds,
        max_backoff_minutes=args.max_backoff_minutes,
        resp_bytes=args.resp_bytes,
        insecure=bool(args.insecure),
        notifier=notifier,
        force_terminal=args.force_terminal,
    )

    console = Console()
    console.print(f"[cyan]ntfy:[/cyan] {ntfy_url}")
    console.print(f"[cyan]Targets:[/cyan] {len(urls)}")
    console.print(f"[cyan]Interval:[/cyan] {args.interval_seconds}s  •  [cyan]Timeout:[/cyan] {args.timeout_seconds}s  •  [cyan]Max backoff:[/cyan] {args.max_backoff_minutes}m  •  [cyan]Heartbeat:[/cyan] {args.heartbeat_hours}h")

    loop = asyncio.get_running_loop()
    for sig in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(asyncio, sig, None) or getattr(__import__("signal"), sig),
                                    monitor.stop)
        except Exception:
            # Not all platforms allow setting signal handlers (e.g., Windows + Python < 3.8)
            pass

    try:
        await monitor.run(heartbeat_hours=args.heartbeat_hours)
    finally:
        console.print("[yellow]Shutting down...[/yellow]")

def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
