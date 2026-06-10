"""Input validation and prompt-injection guardrails."""

import re

from fastapi import HTTPException

MAX_INPUT_LENGTH = 2000

_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"you\s+are\s+now",
    r"jailbreak",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def validate_input(text: str) -> None:
    """Raise HTTP 400 if the text is too long or contains injection patterns."""
    if len(text) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Input exceeds {MAX_INPUT_LENGTH} character limit.",
        )
    for pattern in _COMPILED:
        if pattern.search(text):
            raise HTTPException(
                status_code=400,
                detail="Rejected: prompt injection pattern detected.",
            )
