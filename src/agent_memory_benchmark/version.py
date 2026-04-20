"""Version identifiers.

``__version__`` tracks the package; ``PROTOCOL_VERSION`` tracks the
``MemorySystemShape`` / ``AnswerResult`` contract and only bumps when adapter
authors would need to change their implementation. ``HTTP_API_VERSION`` tracks
the REST shape served by ``HttpAdapter`` (``/v{N}/…``).
"""

from __future__ import annotations

__version__ = "0.1.0"
PROTOCOL_VERSION = "1"
HTTP_API_VERSION = "1"

__all__ = ["__version__", "PROTOCOL_VERSION", "HTTP_API_VERSION"]
