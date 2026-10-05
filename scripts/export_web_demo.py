"""Write the browser scoring bundle and the public holdout sample.

Run from the repo root:

    PYTHONPATH=src python scripts/export_web_demo.py
"""

from telco_nba.portable import write_web_demo_files


def main() -> None:
    for path in write_web_demo_files():
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
