import pandas as pd

from pipeline.validate import build_schema, is_likely_non_english, split_valid_and_quarantined

TIMESTAMP_MIN = pd.Timestamp("1995-01-01")
MIN_WORDS = 5


def _base_row(**overrides) -> dict:
    row = {
        "rating": 5,
        "title": "Great",
        "text": "This album is genuinely wonderful and I recommend it to everyone",
        "asin": "B000000000",
        "parent_asin": "B000000000",
        "user_id": "U1",
        "timestamp": pd.Timestamp("2023-08-15"),
        "helpful_vote": 0,
        "verified_purchase": True,
    }
    row.update(overrides)
    return row


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_valid_row_passes_through():
    df = _make_df([_base_row()])
    schema = build_schema(TIMESTAMP_MIN)

    valid, quarantined = split_valid_and_quarantined(df, schema, MIN_WORDS)

    assert len(valid) == 1
    assert quarantined == []


def test_missing_required_field_is_quarantined():
    df = _make_df([_base_row(user_id=None)])
    schema = build_schema(TIMESTAMP_MIN)

    valid, quarantined = split_valid_and_quarantined(df, schema, MIN_WORDS)

    assert valid.empty
    assert quarantined[0]["reason_code"] == "missing_required_field"


def test_rating_out_of_range_is_quarantined():
    df = _make_df([_base_row(rating=7)])
    schema = build_schema(TIMESTAMP_MIN)

    valid, quarantined = split_valid_and_quarantined(df, schema, MIN_WORDS)

    assert valid.empty
    assert quarantined[0]["reason_code"] == "rating_out_of_range"


def test_empty_text_is_quarantined():
    df = _make_df([_base_row(text="   ")])
    schema = build_schema(TIMESTAMP_MIN)

    valid, quarantined = split_valid_and_quarantined(df, schema, MIN_WORDS)

    assert valid.empty
    assert quarantined[0]["reason_code"] == "empty_text"


def test_implausible_timestamp_is_quarantined():
    df = _make_df([_base_row(timestamp=pd.Timestamp("1990-01-01"))])
    schema = build_schema(TIMESTAMP_MIN)

    valid, quarantined = split_valid_and_quarantined(df, schema, MIN_WORDS)

    assert valid.empty
    assert quarantined[0]["reason_code"] == "implausible_timestamp"


def test_non_english_text_is_quarantined():
    df = _make_df(
        [_base_row(title="Bof", text="Ce CD est vraiment excellent et je le recommande fortement")]
    )
    schema = build_schema(TIMESTAMP_MIN)

    valid, quarantined = split_valid_and_quarantined(df, schema, MIN_WORDS)

    assert valid.empty
    assert quarantined[0]["reason_code"] == "non_english"


def test_short_text_skips_language_check():
    assert is_likely_non_english("Bof", min_words=MIN_WORDS) is False


def test_row_failing_multiple_checks_gets_priority_reason():
    # Fails both rating range and empty text; rating_out_of_range should win
    # per REASON_PRIORITY.
    df = _make_df([_base_row(rating=9, text="")])
    schema = build_schema(TIMESTAMP_MIN)

    _, quarantined = split_valid_and_quarantined(df, schema, MIN_WORDS)

    assert quarantined[0]["reason_code"] == "rating_out_of_range"
