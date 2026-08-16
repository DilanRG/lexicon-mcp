"""Lexicon MCP: an offline multilingual lexical service."""

import os

# Bound the BLAS thread pools before NumPy can initialize them.
#
# NumPy's backends reserve per-thread arenas at load time, scaled to the core
# count. On Windows those reservations count toward commit charge even though
# they are never touched: measured on a 24-core host, importing the runtime
# committed 774 MiB of private bytes against 48 MiB resident. Bounding them
# leaves 34 MiB committed and costs nothing here, because the only in-process
# linear algebra is 300-dimensional dot products during semantic reranking,
# which single-thread anyway -- measured at 11.0 ms against 11.1 ms.
#
# This has to happen before the first NumPy import, so it lives here rather
# than at an entry point. setdefault, so a deliberate caller setting wins.
for _thread_limit in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_limit, "1")
del _thread_limit

__version__ = "1.0.0"
