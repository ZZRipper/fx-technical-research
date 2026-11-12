"""Rebuild and validate the fixed-initial-capital dataset for all Explorer accounts."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fx_simple import write_simple_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced/simple_interest")
    arguments = parser.parse_args()
    summary = write_simple_data(arguments.output_dir, ROOT)
    print(json.dumps({name: summary[name] for name in (
        "status", "accounts", "daily_observations", "compound_reference_max_abs_error_usd",
        "simple_daily_increment_max_abs_error_usd", "newly_reconstructed_v3_trade_ledgers", "payload_bytes",
    )}, indent=2))


if __name__ == "__main__":
    main()
