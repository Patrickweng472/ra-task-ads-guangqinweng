from pathlib import Path

import pandas as pd

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

