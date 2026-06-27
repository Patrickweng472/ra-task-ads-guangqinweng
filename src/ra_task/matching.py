from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


def normalize_company(value: object) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).lower()
    return re.sub(r"[\s\-—_·•,，.。()（）/\\]+", "", text)


def match_companies(
    ads: pd.DataFrame,
    firms: pd.DataFrame,
    aliases_path: Path = Path("config/company_aliases.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match ads to listed firms using exact names and reviewed rules only.

    Fuzzy similarity is deliberately restricted to the review-candidate ledger.
    Ambiguous normalized names or overlapping reviewed rules fail closed.
    """
    firms = firms.copy()
    firms["full_norm"] = firms["公司全称"].map(normalize_company)
    collisions = firms.loc[firms["full_norm"].duplicated(keep=False), ["股票代码", "公司全称", "full_norm"]]
    if len(collisions):
        raise ValueError(f"normalized company names are not unique: {collisions.to_dict('records')}")
    by_full = {name: row for name, (_, row) in zip(firms["full_norm"], firms.iterrows()) if name}
    aliases = pd.read_csv(aliases_path, dtype=str, keep_default_na=False)
    by_code = firms.set_index("股票代码", drop=False)
    choices = dict(zip(firms["股票代码"], firms["公司全称"]))
    rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for _, ad in ads.iterrows():
        names = [ad["公司名称"], ad["关联公司名称"]]
        normalized = [normalize_company(name) for name in names]
        code = ""
        method = "unmatched"
        note = ""
        confidence = "low"
        for name in normalized:
            if name in by_full:
                code = str(by_full[name]["股票代码"])
                method, confidence, note = "exact_normalized", "high", "标准化公司全称精确匹配"
                break
        if not code:
            hits = aliases[aliases.apply(lambda alias: any(re.search(alias["pattern"], name) for name in names), axis=1)]
            hit_codes = set(hits["stock_code"])
            if len(hit_codes) > 1:
                raise ValueError(f"multiple reviewed rules matched ad {ad['canonical_id']}: {sorted(hit_codes)}")
            if len(hit_codes) == 1:
                chosen = hits.sort_values("pattern", key=lambda series: series.str.len(), ascending=False).iloc[0]
                code = chosen["stock_code"]
                if code not in by_code.index:
                    raise ValueError(f"matched reviewed rule references unknown stock code: {code}")
                method = chosen.get("match_method", "reviewed_name_alias") or "reviewed_name_alias"
                confidence, note = "high", chosen["match_note"]
        ranked: dict[str, tuple[str, float]] = {}
        for query in dict.fromkeys(name for name in names if name):
            for candidate_name, score, candidate_code in process.extract(query, choices, scorer=fuzz.WRatio, limit=3):
                if candidate_code not in ranked or score > ranked[candidate_code][1]:
                    ranked[candidate_code] = (candidate_name, float(score))
        ordered = sorted(ranked.items(), key=lambda item: (-item[1][1], item[0]))[:3]
        for rank, (candidate_code, (candidate_name, score)) in enumerate(ordered, start=1):
            candidates.append({"canonical_id": ad["canonical_id"], "candidate_rank": rank, "candidate_stock_code": candidate_code, "candidate_company": candidate_name, "similarity": round(float(score), 2), "accepted": candidate_code == code})
        if code:
            firm = by_code.loc[code]
            stock_name, company_name, industry = firm["证券简称"], firm["公司全称"], firm["证监会行业"]
            status, reason = "matched", ""
        else:
            stock_name = company_name = industry = ""
            status, reason = "unmatched", "无可靠的精确匹配或已复核的母公司/曾用名规则"
        rows.append({
            "canonical_id": ad["canonical_id"], "ad_company": ad["公司名称"], "stock_code": code,
            "stock_name": stock_name, "listed_company": company_name, "industry": industry,
            "match_status": status, "match_method": method, "match_confidence": confidence,
            "match_note": note, "unmatched_reason": reason,
        })
    return pd.DataFrame(rows), pd.DataFrame(candidates)
