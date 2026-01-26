import threading
import time
import subprocess
import re
import logging
import platform
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)

@dataclass
class MemoryStat:
    timestamp: float
    usage_mb: float


class ProcessMemoryMonitor:
    """
    Monitors native process memory usage by process name.
    Works on macOS and Linux.
    """
    def __init__(self, process_name: str, interval: float = 0.5):
        self.process_name = process_name
        self.interval = interval
        self.stats: List[MemoryStat] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.stats = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def get_max_memory(self) -> float:
        if not self.stats:
            return 0.0
        return max(s.usage_mb for s in self.stats)

    def get_avg_memory(self) -> float:
        if not self.stats:
            return 0.0
        return sum(s.usage_mb for s in self.stats) / len(self.stats)

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                mem_mb = self._get_process_memory()
                if mem_mb > 0:
                    self.stats.append(MemoryStat(time.time(), mem_mb))
            except Exception as e:
                pass  # Silently ignore errors
            time.sleep(self.interval)

    def _get_process_memory(self) -> float:
        """Get memory usage of process by name in MB."""
        system = platform.system()

        if system == "Darwin":  # macOS
            # Use ps to get RSS (resident set size) in KB
            cmd = f"ps -A -o rss,comm | grep -i {self.process_name} | head -1 | awk '{{print $1}}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                try:
                    rss_kb = int(result.stdout.strip())
                    return rss_kb / 1024  # Convert to MB
                except ValueError:
                    pass

        elif system == "Linux":
            # Use ps to get RSS in KB
            cmd = f"ps -C {self.process_name} -o rss --no-headers | head -1"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                try:
                    rss_kb = int(result.stdout.strip())
                    return rss_kb / 1024  # Convert to MB
                except ValueError:
                    pass

        return 0.0


class MemoryMonitor:
    """
    Monitors Docker container memory usage in a background thread.
    """
    def __init__(self, container_name: str, interval: float = 0.5):
        self.container_name = container_name
        self.interval = interval
        self.stats: List[MemoryStat] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        
    def start(self):
        self.stats = []
        self._stop_event.clear()
        self._thread.start()
        log.info(f"🧠 Memory monitoring started for container: {self.container_name}")

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        log.info(f"🧠 Memory monitoring stopped. Recorded {len(self.stats)} data points.")

    def get_max_memory(self) -> float:
        if not self.stats:
            return 0.0
        return max(s.usage_mb for s in self.stats)

    def get_avg_memory(self) -> float:
        if not self.stats:
            return 0.0
        return sum(s.usage_mb for s in self.stats) / len(self.stats)

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                # Docker stats command to get memory usage in bytes
                cmd = ["docker", "stats", self.container_name, "--no-stream", "--format", "{{.MemUsage}}"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Output format example: "123MiB / 1GiB" or "1.5GiB / 4GiB"
                    raw = result.stdout.strip().split(" / ")[0]
                    mem_mb = self._parse_memory_str(raw)
                    if mem_mb is not None:
                        self.stats.append(MemoryStat(time.time(), mem_mb))
            except Exception as e:
                log.warning(f"Memory monitor error: {e}")
            
            time.sleep(self.interval)

    def _parse_memory_str(self, mem_str: str) -> float:
        """Converts docker stats string (e.g. '12.5MiB', '1.2GiB') to MB."""
        mem_str = mem_str.strip()
        if not mem_str: return 0.0
        
        # Remove non-numeric chars except dot
        value_str = re.sub(r'[^\d.]', '', mem_str)
        try:
            value = float(value_str)
        except ValueError:
            return 0.0

        if "GiB" in mem_str:
            return value * 1024
        elif "MiB" in mem_str:
            return value
        elif "KiB" in mem_str:
            return value / 1024
        elif "B" in mem_str:
            return value / (1024 * 1024)
        return value
