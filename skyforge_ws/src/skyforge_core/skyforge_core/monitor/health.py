"""Health monitor: hardware faults -> MAYDAY.

Detects motor output saturation (sustained), odometry/camera timeouts,
battery voltage sag and over-temperature. Returns a fault string or None.
Camera loss alone is a degradation (logged), not a MAYDAY: the drone can
still fly home safely without it.
"""
from __future__ import annotations

from typing import Optional, Sequence


class HealthMonitor:
    def __init__(self, params: dict):
        self.hp = params["health"]
        self._sat_since: Optional[float] = None
        self.fault: Optional[str] = None
        self.degraded: set[str] = set()

    def update(
        self,
        t: float,
        motor_outputs: Optional[Sequence[float]] = None,
        last_odom_t: Optional[float] = None,
        last_cam_t: Optional[float] = None,
        cell_voltage: Optional[float] = None,
        temp_c: Optional[float] = None,
    ) -> Optional[str]:
        if self.fault:
            return self.fault  # latched
        hp = self.hp
        if motor_outputs:
            if max(motor_outputs) >= hp["motor_sat_level"]:
                if self._sat_since is None:
                    self._sat_since = t
                elif t - self._sat_since >= hp["motor_sat_s"]:
                    self.fault = f"motor saturation {max(motor_outputs):.2f} for {t - self._sat_since:.1f}s"
            else:
                self._sat_since = None
        if last_odom_t is not None and t - last_odom_t > hp["odom_timeout_s"]:
            self.fault = f"odometry timeout {t - last_odom_t:.2f}s"
        if cell_voltage is not None and cell_voltage < hp["v_min_cell"]:
            self.fault = f"battery voltage sag {cell_voltage:.2f}V/cell"
        if temp_c is not None and temp_c > hp["temp_max_c"]:
            self.fault = f"overheating {temp_c:.0f}C"
        if last_cam_t is not None and t - last_cam_t > hp["cam_timeout_s"]:
            self.degraded.add("camera")
        else:
            self.degraded.discard("camera")
        return self.fault
