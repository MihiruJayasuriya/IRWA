import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["INTERNAL_API_KEY"] = "test-key-for-unit-tests-0123456789"
os.environ.pop("IR_API_KEY", None)
