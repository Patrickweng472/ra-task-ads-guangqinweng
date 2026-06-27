from pathlib import Path

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

