import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

def clean_raw_to_wide_clean(df):
    df = df.copy()

    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    text_cols = [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "results",
    ]
    for col in text_cols:
        if col in df.columns:
            s = df[col].astype("string").str.strip()
            df[col] = s

    judge_pattern = re.compile(r"^week\d+_judge\d+_score$")
    judge_cols = [c for c in df.columns if judge_pattern.match(str(c))]
    for col in judge_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "celebrity_age_during_season" in df.columns:
        df["celebrity_age_during_season"] = pd.to_numeric(
            df["celebrity_age_during_season"], errors="coerce"
        )

    if "celebrity_industry" in df.columns:
        df["celebrity_industry"] = df["celebrity_industry"].fillna("Unknown")
        industries = sorted(df["celebrity_industry"].dropna().astype(str).unique().tolist())
        industry_to_idx = {name: i for i, name in enumerate(industries)}
        df["industry_idx"] = df["celebrity_industry"].astype(str).map(industry_to_idx)
        df["industry_idx"] = pd.to_numeric(df["industry_idx"], errors="coerce").fillna(0).astype(int)

    return df


def build_weekly_long(df, max_weeks=11):
    keep_cols = [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "season",
        "results",
        "placement",
        "youth_factor",
        "is_musician",
        "industry_idx",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    frames = []
    for week in range(1, max_weeks + 1):
        week_cols = [
            f"week{week}_judge1_score",
            f"week{week}_judge2_score",
            f"week{week}_judge3_score",
            f"week{week}_judge4_score",
        ]
        week_cols = [c for c in week_cols if c in df.columns]
        if not week_cols:
            continue

        tmp = df[keep_cols + week_cols].copy()
        for col in week_cols:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

        tmp["week"] = week
        tmp["active_judges_week"] = tmp[week_cols].notnull().sum(axis=1)
        tmp["judge_points"] = tmp[week_cols].fillna(0.0).sum(axis=1)
        tmp["is_active"] = (tmp["judge_points"] > 0).astype(int)
        tmp["eliminated_flag"] = (
            tmp["results"]
            .astype(str)
            .str.contains(f"Eliminated Week {week}", case=False, na=False)
            .astype(int)
        )

        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True)
    long_df["judge_percent"] = (
        long_df.groupby(["season", "week"])["judge_points"]
        .transform(lambda s: s / s.sum() if float(s.sum()) > 0 else 0.0)
        .fillna(0.0)
    )

    return long_df

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "2026_MCM_Problem_C_Data.csv"
    processed_dir = project_root / "data" / "processed"
    processed_path = processed_dir / "processed_mcm_wide_clean.csv"

    processed_dir.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        print(f"Reading data from {raw_path}...")
        df = pd.read_csv(raw_path, encoding="utf-8-sig")
        processed_df = clean_raw_to_wide_clean(df)
        processed_df.to_csv(processed_path, index=False, encoding="utf-8-sig")
        print(f"Data processed and saved to {processed_path}")
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Columns: {processed_df.columns.tolist()}")
    else:
        print(f"Error: Raw data file not found at {raw_path}")