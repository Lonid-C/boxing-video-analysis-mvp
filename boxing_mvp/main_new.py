"""Deprecated compatibility entry point.

Use ``python -m boxing_mvp.main``.  This module is retained so old server
commands do not silently run the obsolete Gemini implementation.
"""

from .main import main


if __name__ == "__main__":
    main()
