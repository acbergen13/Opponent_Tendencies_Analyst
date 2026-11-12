"""Build one full opponent tendencies report from the command line.

Usage::

    python scripts/report.py --opponent BUF --weeks 5 6 7 8 9
    python scripts/report.py --opponent BUF --weeks 5 6 7 8 9 --output report.json

The implementation lives in :mod:`opponent_tendencies.report_cli`; this wrapper
keeps the same ``python scripts/...`` invocation style as the data scripts. A
full report needs VT ARC credentials and the built media index, exactly like the
app. ``--no-cache`` forces a fresh pipeline run.
"""

from __future__ import annotations

from opponent_tendencies.report_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
