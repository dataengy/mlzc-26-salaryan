"""Download source CSV with URL and SHA256 provenance; no API token required."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/foorilla/ai-jobs-net-salaries/main/salaries.csv"


def main(dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=45) as response:
        content = response.read()
    if len(content) < 10_000:
        raise RuntimeError(f"Unexpectedly small download: {len(content)} bytes")
    if b"salary_in_usd" not in content[:3000]:
        raise RuntimeError("CSV header did not contain salary_in_usd")
    dest.write_bytes(content)
    info = {"url": URL, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    (dest.parent / "source.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(info)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, default=Path("data/salaries.csv"))
    main(p.parse_args().dest)
