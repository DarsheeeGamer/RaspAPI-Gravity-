import sys
from pathlib import Path
import uvicorn
src_dir = str(Path(__file__).parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from src.main import app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6969)