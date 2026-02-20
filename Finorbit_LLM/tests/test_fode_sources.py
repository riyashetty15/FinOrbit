from backend.fin_fode.engine.final_output_engine import FinalOutputEngine


def test_fode_appends_sources_when_passages_present():
    eng = FinalOutputEngine()
    out = eng.run(
        {
            "raw_answer": "Hello.",
            "context": {
                "module": "SIP_INVESTMENT",
                "channel": "YOUTH_APP",
                "retrieved_passages": [
                    {"document": "sebi_circular.pdf", "chunk_index": 3, "similarity_score": 0.91},
                    {"document": "sebi_circular.pdf", "chunk_index": 3, "similarity_score": 0.91},  # dup
                    {"document": "factsheet.pdf", "chunk_index": 1, "similarity_score": 0.77},
                ],
            },
        }
    )

    final_answer = out.get("final_answer", "")
    assert "Sources" in final_answer
    assert "sebi_circular.pdf" in final_answer
    assert "factsheet.pdf" in final_answer


def test_fode_does_not_append_sources_when_none_present():
    eng = FinalOutputEngine()
    out = eng.run({"raw_answer": "Hello.", "context": {"module": "SIP_INVESTMENT", "channel": "YOUTH_APP"}})
    final_answer = out.get("final_answer", "")
    assert "Sources" not in final_answer
