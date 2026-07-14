"""Lightweight logging helpers."""


def stage_message(stage: str, message: str) -> str:
    """Format a short stage-prefixed message for examples."""

    return f"[{stage}] {message}"
