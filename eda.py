"""Small reproducible EDA, including target and missing-value plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from model_utils import TARGET, clean_data


def run(source: Path, output: Path) -> None:
    raw = pd.read_csv(source)
    data = clean_data(raw)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    data[TARGET].hist(ax=ax, bins=40)
    ax.set(xlabel="Annual salary (USD)", ylabel="Observations", title="Salary distribution")
    fig.tight_layout()
    fig.savefig(output / "salary_histogram.png", dpi=120)
    plt.close(fig)
    missing = raw.isna().sum().sort_values(ascending=False)
    group = data.groupby("experience_level")[TARGET].median().sort_values()
    lines = [
        "# Exploratory data analysis", "",
        f"Original rows: **{len(raw):,}**; cleaned rows: **{len(data):,}**.",
        f"Target median: **${data[TARGET].median():,.0f}**; 10–90 percentile: "
        f"**${data[TARGET].quantile(.1):,.0f}–${data[TARGET].quantile(.9):,.0f}**.",
        "", "## Missing values (before cleaning)", "",
        *[f"- `{field}`: {int(num)}" for field, num in missing.items()],
        "", "## Median salary by experience level", "",
        *[f"- `{level}`: ${amount:,.0f}" for level, amount in group.items()],
        "", "## Limitations", "",
        "- Survey/sample selection bias; cross-country salary levels may not be comparable.",
        "- No individual identifier; repeated profiles may exist. Exact duplicate records are removed.",
        "- `salary`, `salary_currency`, and `salary_in_usd` are never features.",
        "- These associations are descriptive, not causal.",
    ]
    (output / "EDA.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"EDA written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/salaries.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run(args.data, args.output)
