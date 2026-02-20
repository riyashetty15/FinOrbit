import logging
import re
import os
import asyncio
from openai import OpenAI
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Import RAG tool
# Adjust import path based on actual location. 
# Assuming backend/tools/rag_tool.py exists based on previous file reads.
from backend.tools.rag_tool import knowledge_lookup

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """
    Structured citation with full provenance.
    Represents a single piece of evidence from the knowledge base.
    """
    doc_id: str
    source: str
    page: Optional[int]
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EvidencePack:
    """
    Production-grade evidence package with coverage scoring.
    
    Coverage levels:
    - sufficient: ≥3 verified citations OR ≥2 citations from authoritative doc
    - partial: 1-2 verified citation(s) OR weak relevance
    - insufficient: 0 verified citations
    """
    module: str
    query: str
    citations: List[Citation]
    confidence: float
    coverage: str  # sufficient | partial | insufficient
    filters: Dict[str, Any]
    
    # Optional metadata
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class RetrievalService:
    """
    Service responsible for document retrieval, semantic routing, 
    and evidence verification. Returns a structured EvidencePack.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self._model_name = os.getenv("CUSTOM_MODEL_NAME", "gpt-4o-mini")
        if self.api_key:
             self._openai_client = OpenAI(api_key=self.api_key)
        else:
             self._openai_client = None
             logger.warning("RetrievalService initialized without LLM_API_KEY")

        # Configuration
        self.min_chunk_relevance = 0.45  # Tuned for recall
        self.min_verification_confidence = 0.50
        
        # Fallback Keywords
        self.MODULE_KEYWORDS = {
            "credit": ["credit", "loan", "emi", "cibil", "score", "borrow", "nbfc", "rbi", "lending"],
            "investment": ["invest", "portfolio", "mutual fund", "stock", "equity", "bond", "market", "sebi"],
            "insurance": ["insurance", "premium", "coverage", "term plan", "health insurance", "irdai", "policy"],
            "retirement": ["retire", "pension", "401k", "nps", "retirement planning", "epf", "provident fund"],
            "taxation": ["tax", "itr", "deduction", "exemption", "section", "income tax", "gst", "finance act"]
        }

    def _get_module_from_llm(self, query: str) -> Optional[str]:
        """Use LLM to semantically classify query into a module"""
        if not self._openai_client:
            return None
            
        try:
            valid_modules = list(self.MODULE_KEYWORDS.keys())
            
            prompt = f"""
            Classify the following financial query into exactly one of these categories: 
            {valid_modules}
            
            Query: "{query}"
            
            - "credit": Loans, debt, CIBIL, NBFC, RBI lending rules, credit cards, EMI.
            - "investment": Stocks, mutual funds, gold, bonds, portfolio, markets.
            - "insurance": Life, health, vehicle policies, premiums, claims.
            - "retirement": Pension, NPS, EPF, retirement planning.
            - "taxation": Income tax, GST, deductions, returns.
            
            Return ONLY the category name in lowercase.
            """
            
            response = self._openai_client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return None
                
            module = text.strip().lower()
            
            # Clean up potential extra chars (like 'credit.')
            module = re.sub(r'[^a-z]', '', module)
            
            if module in valid_modules:
                return module
            return None
            
        except Exception as e:
            logger.warning(f"LLM module classification failed: {e}")
            return None

    def determine_module(self, query: str) -> str:
        """Determine which RAG module to query"""
        query_lower = query.lower()
        
        # 1. Try Semantic Classification
        llm_module = self._get_module_from_llm(query)
        if llm_module:
            logger.info(f"Selected module (LLM-Semantic): {llm_module}")
            return llm_module

        # 2. Fallback to Keyword Matching
        module_scores = {}
        for module, keywords in self.MODULE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                module_scores[module] = score
        
        if module_scores:
            module = max(module_scores, key=module_scores.get)
            logger.info(f"Selected module (Keyword): {module}")
            return module
        
        # Default
        logger.info("No clear module match, defaulting to 'investment'")
        return "investment"

    async def verify_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> tuple[List[Citation], float]:
        """
        Verify chunks using LLM to ensure they actually answer the query.
        Returns reliable citations and a confidence score.
        """
        if not chunks:
            return [], 0.0

        # Convert raw chunks to text blob for verification
        context_text = ""
        for i, doc in enumerate(chunks):
            context_text += f"Chunk {i+1}: {doc.get('text', '')}\n\n"

        if not self._openai_client:
            # Fallback if no LLM: trust top 3 chunks
            verified = []
            for doc in chunks[:3]:
                verified.append(Citation(
                   doc_id=str(doc.get("document_id", "unknown")),
                   chunk_id=str(doc.get("id", "unknown")),
                   text=doc.get("text", ""),
                   source=str(doc.get("metadata", {}).get("filename", "unknown")),
                   page=doc.get("metadata", {}).get("page_label") or doc.get("metadata", {}).get("page_number"),
                   score=doc.get("score", 0.5),
                   metadata=doc.get("metadata", {})
                ))
            return verified, 0.5  # Neutral confidence

        try:
            prompt = f"""
            You are a strict relevance verifier for a financial RAG system.
            User Query: "{query}"

            Retrieved Contexts:
            {context_text}

            Task:
            1. Identify which chunks contain information RELEVANT to the query.
            2. Assign a relevance score (0.0 to 1.0) for the whole set.

            Output Format (JSON only):
            {{
                "relevant_chunk_indices": [1, 3],
                "confidence_score": 0.85
            }}
            """
            
            response = self._openai_client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": "You are a relevance verifier. Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            import json
            result_text = (response.choices[0].message.content or "").strip()
            clean_json = result_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_json)
            
            relevant_indices = result.get("relevant_chunk_indices", [])
            confidence = float(result.get("confidence_score", 0.0))
            
            verified_citations = []
            # 1-based index in prompt -> 0-based list
            for idx in relevant_indices:
                list_idx = idx - 1
                if 0 <= list_idx < len(chunks):
                    doc = chunks[list_idx]
                    verified_citations.append(Citation(
                        doc_id=str(doc.get("document_id", "unknown")),
                        chunk_id=str(doc.get("id", "unknown")),
                        text=doc.get("text", ""),
                        source=doc.get("metadata", {}).get("filename", "unknown"),
                        page=doc.get("metadata", {}).get("page_label") or doc.get("metadata", {}).get("page_number"),
                        score=confidence,
                        metadata=doc.get("metadata", {})
                    ))
            
            return verified_citations, confidence

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            # Fallback: Return top k but with low confidence
            fallback = []
            for doc in chunks[:3]:
                fallback.append(Citation(
                    doc_id=str(doc.get("document_id", "unknown")),
                    chunk_id=str(doc.get("id", "unknown")),
                    text=doc.get("text", ""),
                    source=doc.get("metadata", {}).get("filename", "unknown"),
                    page=doc.get("metadata", {}).get("page_label"),
                    score=0.4,
                    metadata=doc.get("metadata", {})
                ))
            return fallback, 0.4

    def _calculate_coverage(self, citations: List[Citation], confidence: float) -> str:
        """
        Calculate evidence coverage based on citation count and quality.
        
        Coverage rules:
        - sufficient: ≥3 verified citations OR ≥2 citations from same authoritative doc
        - partial: 1-2 verified citations OR weak relevance
        - insufficient: 0 verified citations
        """
        if len(citations) == 0:
            return "insufficient"
        
        # Check for multiple citations from same authoritative document
        doc_counts = {}
        for citation in citations:
            doc_id = citation.doc_id
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        # Sufficient if ≥3 citations OR ≥2 from same doc with high confidence
        if len(citations) >= 3:
            return "sufficient"
        
        if any(count >= 2 for count in doc_counts.values()) and confidence >= 0.7:
            return "sufficient"
        
        # Partial if 1-2 citations with decent confidence
        if 1 <= len(citations) <= 2 and confidence >= 0.5:
            return "partial"
        
        # Weak relevance
        if len(citations) > 0 and confidence < 0.5:
            return "partial"
        
        return "insufficient"

    async def retrieve_evidence(
        self, 
        query: str, 
        module: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> EvidencePack:
        """
        Main entry point: Retrieve, Verify, and Package evidence.
        
        Args:
            query: User query text
            module: RAG module to query (credit, taxation, etc.)
            top_k: Number of chunks to retrieve
            filters: Metadata filters (doc_type, year, issuer, jurisdiction, etc.)
        
        Returns:
            EvidencePack with citations and coverage assessment
        """
        
        # 1. Routing
        if not module:
            module = self.determine_module(query)
        
        # Initialize filters dict if None
        if filters is None:
            filters = {}
        
        # 2. Retrieval (Call existing tool)
        # Unpack filters if any
        doc_type = filters.get("doc_type")
        year = filters.get("year") or filters.get("year_min")
        issuer = filters.get("issuer")
        jurisdiction = filters.get("jurisdiction")
        
        logger.info(f"Retrieving from module '{module}' for query: {query} with filters: {filters}")
        
        try:
            rag_output = await knowledge_lookup(
                query=query,
                module=module,
                top_k=top_k,
                doc_type=doc_type,
                year=year,
                issuer=issuer
            )
            
            raw_results = rag_output.get("results", [])
            
            if not rag_output.get("found"):
                return EvidencePack(
                    module=module,
                    query=query,
                    citations=[],
                    confidence=0.0,
                    coverage="insufficient",
                    filters=filters,
                    rejection_reason="No documents found in knowledge base."
                )

            # 3. Verification
            verified_citations, confidence = await self.verify_chunks(query, raw_results)
            
            # 4. Coverage Scoring
            coverage = self._calculate_coverage(verified_citations, confidence)
            
            # 5. Rejection reasoning
            rejection_reason = None
            if coverage == "insufficient":
                rejection_reason = "Retrieved information was not relevant enough to answer specifically."
            elif coverage == "partial":
                rejection_reason = "Only partial evidence available. Answer may be incomplete."
            
            return EvidencePack(
                module=module,
                query=query,
                citations=verified_citations,
                confidence=confidence,
                coverage=coverage,
                filters=filters,
                rejection_reason=rejection_reason
            )

        except Exception as e:
            logger.error(f"Retrieval Service Error: {e}")
            return EvidencePack(
                module=module,
                query=query,
                citations=[],
                confidence=0.0,
                coverage="insufficient",
                filters=filters,
                rejection_reason=f"System error: {str(e)}"
            )
