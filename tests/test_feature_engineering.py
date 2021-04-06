import pytest

def get_net_income_to_total_assets(net_income, total_assets):
    output = net_income/total_assets
    return output

def test_get_net_income_to_total_assets():
    output = get_net_income_to_total_assets(738000, 1000000)
    expected = 0.738
    assert output == expected, \
    """The output should show 0.738."""

def get_cash_flow_to_total_assets(Cash_flow, total_assets):
    output = Cash_flow/total_assets
    return output

def test_get_cash_flow_to_total_assets():
    output = get_cash_flow_to_total_assets(631000, 1000000)
    expected = 0.631
    assert output == expected, \
    """The output should show 0.631"""

def test_feature_engineering():
    expected = [0.738, 0.631]
    total_assets = 1000000
    net_income_to_total_assets= get_net_income_to_total_assets(738000, total_assets)
    cash_flow_to_total_assets = get_cash_flow_to_total_assets(631000, total_assets)
    output = [net_income_to_total_assets, cash_flow_to_total_assets]
    assert output == expected, \
    """The output should show a list, [0.738, 0.631]"""