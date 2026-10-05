import os
import sys

# Make the collector's modules importable when running `pytest` from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("COLLECTOR_API_KEY", "test-key-for-unit-tests-0123456789")
