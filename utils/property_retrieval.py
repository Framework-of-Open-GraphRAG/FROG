import nltk, faiss
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import ngrams


class DBPediaOntologyFaiss:
    def __init__(
        self,
        df_classes: pd.DataFrame,
        df_oproperties: pd.DataFrame,
        df_dproperties: pd.DataFrame,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        # embedding
        self.model_embed = SentenceTransformer(embedding_model_name)
        self.df_classes = df_classes
        self.df_oproperties = df_oproperties
        self.df_dproperties = df_dproperties
        emb_classes = self.model_embed.encode(self.df_classes["classLabel"].tolist())
        emb_oproperties = self.model_embed.encode(
            self.df_oproperties["propertyLabel"].tolist()
        )
        emb_dproperties = self.model_embed.encode(
            self.df_dproperties["propertyLabel"].tolist()
        )

        # indexing
        dimension = emb_classes.shape[1]
        self.index_classes = faiss.IndexFlatL2(dimension)
        self.index_oproperties = faiss.IndexFlatL2(dimension)
        self.index_dproperties = faiss.IndexFlatL2(dimension)
        self.index_classes.add(emb_classes)
        self.index_oproperties.add(emb_oproperties)
        self.index_dproperties.add(emb_dproperties)

        self.stopwords = set(stopwords.words("english"))

    def _search(
        self, index: faiss.IndexFlatL2, df: pd.DataFrame, q: str, k: int = 5
    ) -> pd.DataFrame:
        xq = self.model_embed.encode([q])
        D, I = index.search(xq, k)
        df = df.iloc[I[0]].copy()
        df["sim"] = D[0]
        return df

    def search_classes(self, q: str, k: int = 5) -> pd.DataFrame:
        return self._search(self.index_classes, self.df_classes, q, k)

    def search_oproperties(self, q: str, k: int = 5) -> pd.DataFrame:
        return self._search(self.index_oproperties, self.df_oproperties, q, k)

    def search_dproperties(self, q: str, k: int = 5) -> pd.DataFrame:
        return self._search(self.index_dproperties, self.df_dproperties, q, k)

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
        tokens = self._preprocess_into_tokens(q)
        ngrams = self._generate_ngrams(tokens)
        resources = {
            "classes": (self.index_classes, self.df_classes),
            "objProperties": (self.index_oproperties, self.df_oproperties),
            "dataProperties": (self.index_dproperties, self.df_dproperties),
        }
        result = {"classes": [], "objProperties": [], "dataProperties": []}

        def parallel_search(ngram, name, index, df, threshold=threshold):
            df_res = self._search(index, df, ngram, k)
            return name, df_res[df_res["sim"] < threshold]["short"].tolist()

        with ThreadPoolExecutor() as executor:
            futures = []
            for ngram in ngrams + property_candidates:
                for name, (index, df) in resources.items():
                    futures.append(
                        executor.submit(parallel_search, ngram, name, index, df)
                    )

            for future in futures:
                name, df_res = future.result()
                if df_res:
                    result[name].extend(df_res)
                    result[name] = list(set(result[name]))

        return result
