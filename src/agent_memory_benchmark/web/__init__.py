"""Web dashboard for browsing runs, comparing scorecards, and launching jobs.

Everything in this subpackage is optional — it lives behind the ``[web]``
extra so ``amb run`` users don't pay for fastapi/uvicorn. Import-time
references to fastapi/uvicorn must stay lazy; ``amb serve`` prints a
friendly install hint when the extras are missing rather than crashing
with a ModuleNotFoundError from some transitive import.
"""

from __future__ import annotations

__all__: list[str] = []
