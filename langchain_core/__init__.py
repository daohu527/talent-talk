"""Top-level shim package to satisfy imports of `langchain_core`.

This mirrors the minimal module placed under `src/` so imports work whether the
package is used via the `src` package or as a top-level import during local
development and when running `uvicorn` from the project root.
"""

__all__ = ["messages"]
