from churn_model import ChurnPredictor

def test_high_risk_high_value_gets_discount():
    assert ChurnPredictor.determine_action(0.60, "High") == "Send 10% Discount Offer"
    assert ChurnPredictor.determine_action(0.95, "High") == "Send 10% Discount Offer"


def test_high_risk_other_values_get_support():
    for value in ("Low", "Medium", "high"):  # only the exact string "High" qualifies
        assert ChurnPredictor.determine_action(0.75, value) == "Send Support Message"


def test_medium_and_low_risk_boundaries():
    assert ChurnPredictor.determine_action(0.59, "High") == "Send Personalized Usage Tips"
    assert ChurnPredictor.determine_action(0.40, "Low") == "Send Personalized Usage Tips"
    assert ChurnPredictor.determine_action(0.39, "High") == "No action"


def test_risk_level_matches_action_thresholds():
    assert ChurnPredictor.risk_level(0.60) == "High"
    assert ChurnPredictor.risk_level(0.40) == "Medium"
    assert ChurnPredictor.risk_level(0.39) == "Low"