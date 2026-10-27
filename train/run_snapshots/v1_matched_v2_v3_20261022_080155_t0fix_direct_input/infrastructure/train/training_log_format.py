"""Shared scalar metric formats emitted by the PPO trainers."""
import re

PATTERN = re.compile(
    r"\[u\s+(\d+)\]\s+env=\s*(\d+)\s+rsucc=([\d.]+)"
    r"\s+n_rounds=\s*(\d+)\s+ret=([+\-\d.]+)\s+len=\s*([\d.]+)"
    r"\s+ent\(t0,t>=1\)=\(([\d.]+),([\d.]+)\)"
    r"\s+aL=([+\-\d.]+)\s+vL=([\d.]+).*?(?:switch=([\d.]+))?$"
)
FIELDS = ["update", "env_steps", "round_success", "rounds_completed",
          "episode_return", "episode_steps", "entropy_t0", "entropy_movement",
          "actor_loss", "value_loss", "assignment_switch_rate"]

LOG_LINE = re.compile(
    r"^\[u\s+(\d+)\]\s+env=\s*(\d+)\s+rsucc=([\d.]+)"
    r"\s+n_rounds=\s*(\d+)\s+ret=([+\-\d.]+)\s+len=\s*([\d.]+)\s"
)
