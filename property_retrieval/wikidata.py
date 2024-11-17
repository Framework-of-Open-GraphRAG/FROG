import faiss
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from property_retrieval.base import BasePropertyRetrieval


class WikidataPropertyRetrieval(BasePropertyRetrieval):
    def __init__(
        self,
        df_properties: pd.DataFrame,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        super().__init__(embedding_model_name)
        self.df_properties = df_properties
        emb_properties = self.model_embed.encode(
            self.df_properties["propertyLabel"].tolist()
        )

        dimension = emb_properties.shape[1]
        self.index_properties = faiss.IndexFlatL2(dimension)
        self.index_properties.add(emb_properties)

    def search_properties(self, q: str, k: int = 5) -> pd.DataFrame:
        return self._search(self.index_properties, self.df_properties, q, k)

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
            "properties": (self.index_properties, self.df_properties),
        }
        result = {"properties": []}

        def parallel_search(ngram, name, index, df, threshold=threshold):
            df_res = self._search(index, df, ngram, k)
            df_res["idWithLabel"] = (
                df_res["propertyId"] + " - " + df_res["propertyLabel"]
            )
            return (
                name,
                df_res[df_res["sim"] < threshold]["idWithLabel"].tolist(),
            )

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
