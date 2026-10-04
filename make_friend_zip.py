"""Build a portable project ZIP without local environments or private keys."""

from pathlib import Path
import os
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Waterwise_All_Agents.zip"
SKIP_DIRS = {"venv", ".venv", "__pycache__", ".git", "logs", ".pytest_cache"}
SKIP_FILES = {".env", "historical_usage.db", OUTPUT.name}

with ZipFile(OUTPUT, "w", ZIP_DEFLATED, compresslevel=6) as archive:
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        for name in files:
            path = Path(folder) / name
            if name in SKIP_FILES or path.suffix in {".pyc", ".log"}:
                continue
            archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))

    # Every service reads this file. The friend package gets local demo values,
    # while the private configuration in this workspace stays out of the ZIP.
    example = (ROOT / "router-agent" / ".env.example").read_bytes()
    archive.writestr(f"{ROOT.name}/router-agent/.env", example)

print(f"Created {OUTPUT.name} ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MB)")
