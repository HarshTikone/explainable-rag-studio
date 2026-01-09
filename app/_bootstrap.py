import os
import sys

def bootstrap():
    """
    Ensures the project root is on sys.path so imports like `from backend...` work
    regardless of how Streamlit launches the script.
    """
    # app/ is one level below the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
