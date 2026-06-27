from pathlib import Path

import pandas as pd
import pytest

from ra_task.matching import match_companies, normalize_company


def test_normalize_company_handles_parentheses() -> None:
    assert normalize_company("胜宏科技(惠州)股份有限公司") == normalize_company("胜宏科技惠州股份有限公司")


def test_parent_rules_do_not_confuse_ping_an_bank() -> None:
    ads = pd.DataFrame([
        {"canonical_id": "1", "公司名称": "中国平安人寿保险股份有限公司上海分公司", "关联公司名称": "中国平安人寿保险股份有限公司上海分公司"},
        {"canonical_id": "2", "公司名称": "招商银行股份有限公司厦门分行", "关联公司名称": "招商银行股份有限公司厦门分行"},
        {"canonical_id": "3", "公司名称": "上海万科物业服务有限公司", "关联公司名称": "上海万科物业服务有限公司"},
    ])
    firms = pd.DataFrame([
        {"股票代码": "000001.SZ", "证券简称": "平安银行", "公司全称": "平安银行股份有限公司", "证监会行业": "金融业"},
        {"股票代码": "601318.SH", "证券简称": "中国平安", "公司全称": "中国平安保险(集团)股份有限公司", "证监会行业": "金融业"},
        {"股票代码": "600036.SH", "证券简称": "招商银行", "公司全称": "招商银行股份有限公司", "证监会行业": "金融业"},
        {"股票代码": "000002.SZ", "证券简称": "万科A", "公司全称": "万科企业股份有限公司", "证监会行业": "房地产业"},
    ])
    matches, _ = match_companies(ads, firms, Path("config/company_aliases.csv"))
    assert matches["stock_code"].tolist() == ["601318.SH", "600036.SH", "000002.SZ"]
    assert set(matches["match_method"]) == {"reviewed_parent_rule"}


def test_reviewed_aliases_fix_and_avoid_real_false_positives() -> None:
    ads = pd.DataFrame([
        {"canonical_id": "58", "公司名称": "岭南园林股份有限公司", "关联公司名称": "岭南园林股份有限公司"},
        {"canonical_id": "298", "公司名称": "中关村科技租赁股份有限公司", "关联公司名称": "中关村科技租赁股份有限公司"},
    ])
    firms = pd.DataFrame([
        {"股票代码": "002717.SZ", "证券简称": "*ST岭南", "公司全称": "岭南生态文旅股份有限公司", "证监会行业": "土木工程建筑业"},
        {"股票代码": "605303.SH", "证券简称": "园林股份", "公司全称": "杭州市园林绿化股份有限公司", "证监会行业": "土木工程建筑业"},
        {"股票代码": "000931.SZ", "证券简称": "中关村", "公司全称": "北京中关村科技发展(控股)股份有限公司", "证监会行业": "医药制造业"},
    ])
    matches, candidates = match_companies(ads, firms, Path("config/company_aliases.csv"))
    assert matches.loc[0, "stock_code"] == "002717.SZ"
    assert matches.loc[0, "match_method"] == "reviewed_name_alias"
    assert matches.loc[1, "match_status"] == "unmatched"
    assert not candidates.loc[candidates["canonical_id"].eq("298"), "accepted"].any()


def test_overlapping_review_rules_fail_instead_of_using_file_order(tmp_path: Path) -> None:
    ads = pd.DataFrame([{"canonical_id": "1", "公司名称": "甲科技股份有限公司", "关联公司名称": "甲科技股份有限公司"}])
    firms = pd.DataFrame([
        {"股票代码": "000001.SZ", "证券简称": "甲", "公司全称": "甲集团股份有限公司", "证监会行业": "A"},
        {"股票代码": "000002.SZ", "证券简称": "乙", "公司全称": "乙集团股份有限公司", "证监会行业": "B"},
    ])
    aliases = tmp_path / "aliases.csv"
    pd.DataFrame([
        {"pattern": "甲科技", "stock_code": "000001.SZ", "match_method": "reviewed_name_alias", "match_note": "one"},
        {"pattern": "甲科技股份", "stock_code": "000002.SZ", "match_method": "reviewed_name_alias", "match_note": "two"},
    ]).to_csv(aliases, index=False)
    with pytest.raises(ValueError, match="multiple reviewed rules"):
        match_companies(ads, firms, aliases)


def test_all_reviewed_rules_have_provenance() -> None:
    aliases = pd.read_csv("config/company_aliases.csv", dtype=str, keep_default_na=False)
    required = ["reviewer", "reviewed_at", "review_basis"]
    assert set(required).issubset(aliases.columns)
    assert not aliases[required].eq("").any().any()
