"""Small reproducible EDA, including target and missing-value plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import get_config, project_path
from .model_utils import FEATURES, TARGET, clean_data, make_model
from .train import split_data

CONFIG = get_config()


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
    model_config = CONFIG["model"]
    model_data = data.sample(n=model_config["max_rows"], random_state=model_config["sample_seed"]).reset_index(drop=True) if len(data) > model_config["max_rows"] else data
    train, validation, test, split_info = split_data(model_data)
    importance_config = CONFIG["eda"]
    model = make_model(importance_config["importance_model"], importance_config["importance_param"])
    fit_data = pd.concat([train, validation], ignore_index=True)
    model.fit(fit_data[FEATURES], fit_data[TARGET])
    importance_data = test.sample(n=min(importance_config["importance_max_rows"], len(test)), random_state=importance_config["importance_seed"])
    importance = permutation_importance(
        model,
        importance_data[FEATURES],
        importance_data[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=importance_config["importance_repeats"],
        random_state=importance_config["importance_seed"],
        n_jobs=importance_config["permutation_jobs"],
    )
    importance_frame = pd.DataFrame({
        "feature": FEATURES,
        "mae_increase_usd": importance.importances_mean,
        "std_usd": importance.importances_std,
    }).sort_values("mae_increase_usd", ascending=False)
    ordered = importance_frame.sort_values("mae_increase_usd")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(ordered["feature"], ordered["mae_increase_usd"], xerr=ordered["std_usd"])
    ax.set(xlabel="Increase in held-out MAE when shuffled (USD)", title="Permutation feature importance")
    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=120)
    plt.close(fig)
    importance_frame.to_csv(output / "feature_importance.csv", index=False)
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
        "", "## Feature importance", "",
        f"Permutation importance uses Ridge(alpha=1), the `{split_info}` split, and {len(importance_data):,} held-out rows with two shuffles per feature. Values show the increase in MAE when each feature is shuffled; larger positive values indicate greater predictive contribution for this fitted model.",
        "",
        *[f"- `{row.feature}`: MAE increase ${row.mae_increase_usd:,.0f} ± ${row.std_usd:,.0f}" for row in importance_frame.itertuples()],
        "",
        "These values describe predictive association, not causation. Correlated features can share or mask importance.",
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
    parser.add_argument("--data", type=Path, default=project_path("dataset"))
    parser.add_argument("--output", type=Path, default=project_path("reports"))
    args = parser.parse_args()
    run(args.data, args.output)
