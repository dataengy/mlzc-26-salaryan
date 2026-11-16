"""Download source CSV with URL and SHA256 provenance; no API token required."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from mlzc_26_salaryan.config import get_config, project_path

CONFIG = get_config()
URL = CONFIG["data"]["source_url"]


def main(dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=45) as response:
        content = response.read()
    if len(content) < CONFIG["data"]["minimum_download_bytes"]:
        raise RuntimeError(f"Unexpectedly small download: {len(content)} bytes")
    if CONFIG["data"]["target"].encode() not in content[:3000]:
        raise RuntimeError("CSV header did not contain salary_in_usd")
    dest.write_bytes(content)
    info = {"url": URL, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    provenance = project_path("provenance") if dest == project_path("dataset") else dest.parent / "source.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(info)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, default=project_path("dataset"))
    main(p.parse_args().dest)
