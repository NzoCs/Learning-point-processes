"""The automated test suite renders plots without a desktop display."""

import os

os.environ["MPLBACKEND"] = "Agg"
