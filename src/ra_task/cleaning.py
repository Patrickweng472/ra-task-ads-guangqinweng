from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

TOKEN_RE = re.compile(r"<\$&\d+&\$>")
VALID_TICKER_RE = re.compile(r"^\d{6}\.(?:SZ|SH|BJ)$")
AD_COLUMNS = ["id", "公司名称", "关联公司名称", "岗位", "岗位描述", "岗位标签", "待遇", "学历", "所在城市", "发布时间"]


def clean_text(value: object, *, tag_field: bool = False) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    replacement = "; " if tag_field else " "
    text = TOKEN_RE.sub(replacement, text).replace("\u00a0", " ")
    if tag_field:
        text = re.sub(r"(?:\s*;\s*)+", "; ", text).strip(" ;")
    return re.sub(r"\s+", " ", text).strip()


def clean_ads(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    ads = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if list(ads.columns) != AD_COLUMNS:
        raise ValueError(f"Unexpected ad columns: {list(ads.columns)}")
    raw_rows = len(ads)
    for column in AD_COLUMNS[1:]:
        ads[column] = ads[column].map(lambda value, c=column: clean_text(value, tag_field=c == "岗位标签"))
    parsed = pd.to_datetime(ads["发布时间"], format="mixed", errors="coerce")
    if parsed.isna().any():
        bad = ads.loc[parsed.isna(), ["id", "发布时间"]].to_dict("records")
        raise ValueError(f"Unparseable dates: {bad}")
    ads["published_date"] = parsed.dt.strftime("%Y-%m-%d")
    ads["year"] = parsed.dt.year.astype(int)
    business = AD_COLUMNS[1:]
    group_key = ads.groupby(business, dropna=False, sort=False).ngroup()
    ads["duplicate_group"] = group_key
    source_ids = ads.groupby("duplicate_group", sort=False)["id"].agg(lambda s: "|".join(s.astype(str)))
    sizes = ads.groupby("duplicate_group", sort=False).size()
    canonical = ads.drop_duplicates(business, keep="first").copy()
    canonical["canonical_id"] = canonical["id"]
    canonical["source_ids"] = canonical["duplicate_group"].map(source_ids)
    canonical["duplicate_group_size"] = canonical["duplicate_group"].map(sizes).astype(int)
    canonical = canonical.drop(columns=["duplicate_group"])
    canonical = canonical[["canonical_id", "source_ids", "duplicate_group_size", *AD_COLUMNS[1:], "published_date", "year"]]
    id_to_canonical = {}
    for _, row in canonical.iterrows():
        for source_id in row["source_ids"].split("|"):
            id_to_canonical[source_id] = row["canonical_id"]
    duplicate_map = pd.DataFrame(
        [{"source_id": source, "canonical_id": canon, "is_duplicate": source != canon} for source, canon in id_to_canonical.items()]
    ).sort_values("source_id", key=lambda s: s.astype(int))
    stats = {
        "raw_ads": raw_rows,
        "canonical_ads": len(canonical),
        "duplicates_removed": raw_rows - len(canonical),
        "duplicate_groups": int((canonical["duplicate_group_size"] > 1).sum()),
    }
    return canonical, duplicate_map, stats


def clean_firms(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    firms = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    raw_rows = len(firms)
    firms = firms[firms["股票代码"].str.match(VALID_TICKER_RE, na=False)].copy()
    for column in ["证券简称", "公司全称", "证监会行业"]:
        firms[column] = firms[column].map(clean_text)
    if firms["股票代码"].duplicated().any():
        raise ValueError("Duplicate stock codes in firm master")
    return firms, {"raw_firm_rows": raw_rows, "valid_firms": len(firms), "invalid_firm_rows": raw_rows - len(firms)}

