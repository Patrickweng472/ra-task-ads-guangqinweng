from pathlib import Path

import pandas as pd

from ra_task.cleaning import TOKEN_RE, clean_ads, clean_firms, clean_text


def test_clean_text_removes_tokens_and_normalizes() -> None:
    assert clean_text("Ａ<$&0005&$><$&0006&$>  B\u00a0") == "A B"
    assert clean_text("数据<$&0007&$>AI<$&0005&$>", tag_field=True) == "数据; AI"


def test_real_data_acceptance_counts() -> None:
    ads, duplicate_map, stats = clean_ads(Path("data/raw/ra_task_ads.csv"))
    firms, firm_stats = clean_firms(Path("data/raw/ra_task_firms.csv"))
    assert stats == {"raw_ads": 612, "canonical_ads": 573, "duplicates_removed": 39, "duplicate_groups": 38}
    assert len(duplicate_map) == 612
    assert firm_stats["valid_firms"] == 5461
    assert not ads["岗位描述"].str.contains(TOKEN_RE).any()
    assert set(ads["year"]) == set(range(2014, 2026))
    assert firms["股票代码"].is_unique


def test_exact_duplicates_are_removed_but_near_duplicates_remain(tmp_path: Path) -> None:
    base = {"公司名称": "甲公司", "关联公司名称": "甲公司", "岗位": "数据分析", "岗位描述": "分析数据", "岗位标签": "SQL", "待遇": "10k", "学历": "本科", "所在城市": "北京", "发布时间": "2025-01-02 08:00:00"}
    rows = [{"id": "1", **base}, {"id": "2", **base}, {"id": "3", **{**base, "岗位描述": "分析数据。", "发布时间": "2025-01-03"}}]
    path = tmp_path / "ads.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    cleaned, mapping, stats = clean_ads(path)
    assert len(cleaned) == 2
    assert stats["duplicates_removed"] == 1
    assert mapping.loc[mapping["source_id"].eq("2"), "canonical_id"].item() == "1"
    assert set(cleaned["published_date"]) == {"2025-01-02", "2025-01-03"}
