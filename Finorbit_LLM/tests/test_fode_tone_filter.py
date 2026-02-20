from backend.fin_fode.engine.tone_filter import ToneFilter


def test_tone_filter_does_not_weaken_never_for_fraud_module():
    tf = ToneFilter({
        "force_disclaimer": False,
        "banned_phrases": ["never"],
        "replacements": {"never": "rarely"},
    })

    text = "Never share your OTP."
    out, applied, flags, meta = tf.apply(text, {"module": "FRAUD"})

    assert "Never share" in out
    assert "rarely" not in out.lower()
