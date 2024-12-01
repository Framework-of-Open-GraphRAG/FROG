import pandas as pd
import weaviate.classes as wvc


from property_retrieval.base import BasePropertyRetrieval


class WikidataPropertyRetrieval(BasePropertyRetrieval):
    def __init__(
        self,
        df_properties: pd.DataFrame,
        embedding_model_name: str = "jinaai/jina-embeddings-v3",
        is_local_client: bool = True,
    ) -> None:
        super().__init__(
            db_collection_name="wikidata_property_db",
            embedding_model_name=embedding_model_name,
            is_local_client=is_local_client,
        )
        self.df_properties = df_properties

        if self.is_collection_empty:
            emb_properties = self.model_embed.encode(
                self.df_properties["label"].tolist(), show_progress_bar=True
            )

            with self.collection.batch.dynamic() as batch:
                for i, row in df_properties.iterrows():
                    batch.add_object(
                        properties=row.to_dict(),
                        vector=emb_properties[i].tolist(),
                    )

    def search_properties(self, q: str, k: int = 5) -> pd.DataFrame:
        return self._search(q, k=k)

    def get_related_candidates(
        self,
        q: str,
        property_candidates: list[str] = [],
        threshold: int = 0.5,
        k: int = 5,
    ) -> dict[str, list[str]]:
        tokens = self._preprocess_into_tokens(q)
        ngrams = self._generate_ngrams(tokens)
        result = {"properties": []}

        def search(ngram, type, threshold=threshold):
            df_res = self._search(ngram, k=k)
            df_res["idWithLabel"] = df_res["propertyId"] + " - " + df_res["label"]
            return (
                type,
                df_res[df_res["score"] >= threshold]["idWithLabel"].tolist(),
            )

        for ngram in ngrams + property_candidates:
            for type in result.keys():
                type, df_res = search(ngram, type)
                if df_res:
                    result[type].extend(df_res)
                    result[type] = list(set(result[type]))

        return result
