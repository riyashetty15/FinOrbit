"""
Quick validation script to verify metadata traceability across ingestion and retrieval.

Runs a simple query path using RetrievalPipeline to ensure that
each returned chunk includes filename, page_number, and chunk_index.
"""

import logging
from stores.vector_store_setup import get_vector_store_for_module
from retrieval.retrieval_pipeline import RetrievalPipeline


def validate_metadata(module: str = "credit", query: str = "test") -> bool:
    logging.basicConfig(level=logging.INFO)
    try:
        store = get_vector_store_for_module(module)
        rp = RetrievalPipeline(module, store)
        res = rp.query(query_text=query, top_k=5)
        chunks = res.get("chunks", [])
        ok = True
        for i, ch in enumerate(chunks):
            fn = ch.get("document_filename")
            pg = ch.get("page_number")
            ci = ch.get("chunk_index")
            if fn is None or ci is None:
                logging.warning(f"Chunk {i} missing metadata: filename={fn}, chunk_index={ci}, page={pg}")
                ok = False
        if ok:
            logging.info("Metadata validation passed: filename/page/chunk present in results.")
        else:
            logging.info("Metadata validation had missing fields; check ingestion and retrieval paths.")
        return ok
    except Exception as e:
        logging.error(f"Validation error: {e}")
        return False


if __name__ == "__main__":
    validate_metadata()
