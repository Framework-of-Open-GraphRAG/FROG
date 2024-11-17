import nltk, faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import ngrams


class BasePropertyRetrieval:
    def __init__(
        self, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ) -> None:
        self.model_embed = SentenceTransformer(embedding_model_name)
        self.stopwords = set(stopwords.words("english"))

    def _search(
        self, index: faiss.IndexFlatL2, df: pd.DataFrame, q: str, k: int = 5
    ) -> pd.DataFrame:
        xq = self.model_embed.encode([q])
        D, I = index.search(xq, k)
        df = df.iloc[I[0]].copy()
        df["sim"] = D[0]
        return df

    def _preprocess_into_tokens(self, q: str) -> list[str]:
        tok_pattern = r"\w+"
        tokenizer = RegexpTokenizer(tok_pattern)
        tokenized = tokenizer.tokenize(q)
        result = []
        for tok in tokenized:
            tok = tok.lower()
            if tok not in self.stopwords:
                result.append(tok)
        return result

    def _generate_ngrams(self, tokens: list[str]) -> list[str]:
        max_n = len(tokens)
        result = []
        for n in range(1, max_n + 1):
            n_grams = ngrams(tokens, n)
            result.extend([" ".join(ng) for ng in n_grams])
        return result

    def get_related_candidates(
        self,
        q: str,
        property_candidates: list[str] = [],
        threshold: int = 0.5,
        k: int = 5,
    ) -> dict[str, list[str]]:
        raise NotImplementedError("This method should be overridden by subclasses")
