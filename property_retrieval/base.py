import nltk
import pandas as pd
from sentence_transformers import SentenceTransformer

import weaviate
import weaviate.classes as wvc

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import ngrams


class BasePropertyRetrieval:
    def __init__(
        self,
        db_collection_name: str,
        embedding_model_name: str = "jinaai/jina-embeddings-v3",
    ) -> None:
        self.model_embed = SentenceTransformer(
            embedding_model_name, trust_remote_code=True
        )
        self.client = weaviate.connect_to_local(skip_init_checks=True)
        if not self.client.collections.exists(db_collection_name):
            self.collection = self.client.collections.create(
                name=db_collection_name,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            )
            self.is_collection_empty = True
        else:
            self.collection = self.client.collections.get(db_collection_name)
            self.is_collection_empty = False
        self.stopwords = set(stopwords.words("english"))

    def _search(self, q: str, type: str = None, k: int = 5) -> pd.DataFrame:
        query_vector = self.model_embed.encode([q])[0]
        response = self.collection.query.hybrid(
            query=q,
            query_properties=["label"],
            vector=query_vector,
            filters=wvc.query.Filter.by_property("type").equal(type) if type else None,
            return_metadata=wvc.query.MetadataQuery(score=True),
            limit=k,
        )
        df = pd.DataFrame(
            [{**o.properties, "score": o.metadata.score} for o in response.objects]
        )
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
