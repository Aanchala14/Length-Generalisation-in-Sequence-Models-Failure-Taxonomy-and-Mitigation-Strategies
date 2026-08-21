from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "outputs/analysis/failure_taxonomy/failure_taxonomy.csv"
)

OUTPUT_DIR = Path(
    "outputs/analysis/failure_taxonomy"
)

OUTPUT_PATH = OUTPUT_DIR / "failure_taxonomy_compact.csv"


def main():
    taxonomy = pd.read_csv(INPUT_PATH)

    compact = taxonomy[
        [
            "Failure Type",
            "Tasks Affected",
            "Evidence",
            "Interpretation",
        ]
    ]

    compact.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(compact.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()