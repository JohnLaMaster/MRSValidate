"""Shared pytest configuration.

Forces a non-interactive Matplotlib backend before any test imports
pyplot, so the suite runs headlessly (CI, no display) without popping
windows or failing on a missing display server.
"""

import matplotlib

matplotlib.use("Agg")
