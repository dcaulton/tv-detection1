# /classes/entity_resolvers.py
import os
from typing import Dict, Optional, Tuple

import wikipedia
from rapidfuzz import fuzz, process
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama   # swap for your SLM backend
from langchain_core.language_models import BaseLanguageModel

class EntityResolver:
    """
    Handles mapping local media item → best Wikipedia (and later Fandom) page.
    Designed to be instantiated with configurable LLM, thresholds, etc.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        min_confidence: float = 0.75,
        max_search_results: int = 5,
    ):
        self.llm = llm
        self.min_confidence = min_confidence
        self.max_search_results = max_search_results

    def _basic_wikipedia_search(
        self, title: str, year: Optional[int], media_type: str = "movie"
    ) -> Tuple[Optional[str], float]:
        """Fast path: API search + fuzzy matching, no LLM."""
        search_term = f"{title} ({year} {media_type})" if year else title
        try:
            results = wikipedia.search(search_term, results=self.max_search_results)
            if not results:
                return None, 0.0

            # Rank by fuzzy token sort (good for title variations)
            scorer = lambda x: fuzz.token_sort_ratio(search_term.lower(), x.lower())
            best_match, best_score = process.extractOne(
                search_term, results, scorer=scorer
            )

            if best_score < self.min_confidence * 100:
                return None, best_score / 100

            page = wikipedia.page(best_match)
            return page.url, best_score / 100

        except Exception as e:
            print(f"Wikipedia lookup failed: {e}")
            return None, 0.0

    def _llm_disambiguate(
        self, candidates: list[str], item: Dict
    ) -> Optional[str]:
        """LLM tie-breaker when multiple plausible matches."""
        prompt = ChatPromptTemplate.from_template(
            """You are a precise media metadata expert.
Given item: {title} ({year or 'unknown'}) - {type}
Possible Wikipedia page titles:
{candidates}

Return ONLY the single best-matching page title exactly as listed,
or "NONE" if none are a good fit (confidence < ~80%)."""
        )

        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke(
            {
                "title": item.get("title", ""),
                "year": item.get("year", "unknown"),
                "type": item.get("type", "unknown"),
                "candidates": "\n".join(f"- {c}" for c in candidates),
            }
        )

        chosen = response.strip()
        if chosen.upper() == "NONE":
            return None

        try:
            page = wikipedia.page(chosen)
            return page.url
        except:
            return None

    def resolve(self, item: Dict) -> Dict:
        """
        Main entry point.
        item example: {"title": "Caddyshack", "year": 1980, "type": "movie", ...}
        Returns: {"wiki_url": str|None, "confidence": float, "method": str, ...}
        """
        title = item.get("title")
        if not title:
            return {"wiki_url": None, "confidence": 0.0, "method": "invalid_input"}

        url, conf = self._basic_wikipedia_search(
            title, item.get("year"), item.get("type", "movie")
        )

        if url and conf >= self.min_confidence:
            return {
                "wiki_url": url,
                "confidence": conf,
                "method": "wikipedia_basic_fuzzy",
            }

        # Optional LLM step if basic failed or low conf (toggleable later)
        # For now: return what we have
        return {
            "wiki_url": url,
            "confidence": conf,
            "method": "wikipedia_basic_fuzzy",
        }


# For dev testing in src/app.py or jupyter
if __name__ == "__main__":
    # Example instantiation (swap model as needed)
    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.0)

    resolver = EntityResolver(llm=llm, min_confidence=0.75)

    sample = {
        "title": "Caddyshack",
        "year": 1980,
        "type": "movie",
    }

    result = resolver.resolve(sample)
    print(result)
