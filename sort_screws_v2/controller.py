from atexit import register
from time import time
from typing import Literal

from serial import Serial


class Controller(object):
    def __init__(self, port: str, *, baud_rate: int = 9600) -> None:
        self.port: str = port
        self._serial: Serial = Serial(port, baud_rate)
        register(self._serial.close)
        self.wait_for("Ready.")

    def wait_for(self, pattern: str, *, timeout: float = 10, error_when_timeout: bool = False) -> tuple[
        bool, list[str]]:
        start = time()
        buffer = []
        while time() - start < timeout:
            line = self._serial.readline().decode("utf8", errors="ignore").strip()
            if line:
                buffer.append(line)
                if pattern in line:
                    return True, buffer
        if error_when_timeout:
            raise TimeoutError(f"Timed out waiting for Arduino to send \"{pattern}\", received: {buffer}")
        return False, buffer

    def send_command(self, command: str) -> str | None:
        self._serial.write((command.strip() + "\n").encode("utf8"))
        self._serial.flush()
        ok, buffer = self.wait_for("degrees.")
        return buffer[-1] if ok else None

    def turn_to(self, device_id: Literal["A", "B"], angle: int) -> bool:
        return self.send_command(f"turn {device_id} {angle}") is not None

    def reset(self, device_id: Literal["A", "B"]) -> bool:
        return self.send_command(f"reset {device_id.lower()}") is not None
