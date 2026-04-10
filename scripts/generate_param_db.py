#!/usr/bin/env python3
"""
Generate param.json and param_id.json from the ECMWF parameter database API.

Usage:
    python scripts/generate_param_db.py [--output-dir DIR] [--provider PROVIDER]

Outputs:
    <output-dir>/param.json    - dict keyed by string param id, value is the full param object
    <output-dir>/param_id.json - dict keyed by shortname, value is the string param id

When multiple API entries share the same shortname the last occurrence (by
position in the API response, i.e. the entry with the highest id) is used,
which matches the convention of the existing files.
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

API_URL = "https://codes.ecmwf.int/parameter-database/api/v1/param/?format=json"

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "covjsonkit" / "data" / "ecmwf"


def fetch_params(url: str) -> list:
    """Fetch all parameters from the API endpoint."""
    print(f"Fetching parameters from {url} ...")
    try:
        with urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        print(f"ERROR: Failed to fetch data from API: {exc}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse API response as JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print("ERROR: Expected a JSON array from the API.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetched {len(data)} parameter entries.")
    return data


def build_param_json(params: list) -> dict:
    """Build param.json: {str(id): full_param_object}."""
    return {str(p["id"]): p for p in params}


def build_param_id_json(params: list) -> dict:
    """
    Build param_id.json: {shortname: str(id)}.

    For shortnames that appear more than once, the last occurrence in the
    API response is used (matching the convention of the existing files).
    """
    result = {}
    for p in params:
        shortname = p["shortname"]
        result[shortname] = str(p["id"])
    return result


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Written {len(data)} entries to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write param.json and param_id.json (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--url",
        default=API_URL,
        help=f"API endpoint URL (default: {API_URL})",
    )
    args = parser.parse_args()

    params = fetch_params(args.url)

    param_json = build_param_json(params)
    param_id_json = build_param_id_json(params)

    write_json(args.output_dir / "param.json", param_json)
    write_json(args.output_dir / "param_id.json", param_id_json)

    print("Done.")


if __name__ == "__main__":
    main()
