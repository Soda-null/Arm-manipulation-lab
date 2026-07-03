"""Plotting helper skeletons for project visualizations."""

from __future__ import annotations

from typing import Sequence


def plot_arm(q: Sequence[float], link_lengths: Sequence[float], output_path: str) -> None:
    """Plot a planar arm configuration."""
    raise NotImplementedError("Plotting implementation pending.")


def plot_workspace(points: Sequence[Sequence[float]], output_path: str) -> None:
    """Plot reachable workspace samples."""
    raise NotImplementedError("Plotting implementation pending.")


def plot_trajectory(points: Sequence[Sequence[float]], output_path: str) -> None:
    """Plot an end-effector trajectory."""
    raise NotImplementedError("Plotting implementation pending.")
