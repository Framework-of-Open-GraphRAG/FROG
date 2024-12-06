import pandas as pd
from rdflib import Graph

from property_retrieval.base import BasePropertyRetrieval


class EnterprisePropertyRetrieval(BasePropertyRetrieval):
    def __init__(
        self,
        turtle_file_path: str,
        get_entities_query: str,
        get_properties_query: str,
        embedding_model_name: str = "jinaai/jina-embeddings-v3",
        is_local_client: bool = True,
    ) -> None:
        super().__init__(
            db_collection_name="enterprise_property_db",
            embedding_model_name=embedding_model_name,
            is_local_client=is_local_client,
        )

        if self.is_collection_empty:
            g = Graph().parse(turtle_file_path)
            entity_response = g.query(get_entities_query)
            property_response = g.query(get_properties_query)
            df_entities = pd.DataFrame(entity_response.bindings)
            df_properties = pd.DataFrame(property_response.bindings)
            df_entities.columns = [str(col) for col in df_entities.columns]
            df_properties.columns = [str(col) for col in df_properties.columns]
            for col in df_entities.columns:
                df_entities[col] = df_entities[col].apply(lambda x: str(x))

            emb_entities = self.model_embed.encode(
                df_entities["label"].tolist(), show_progress_bar=True
            )
            emb_properties = self.model_embed.encode(
                df_properties["label"].tolist(), show_progress_bar=True
            )
            enterprise_df_vectors = {
                "entities": (df_entities, emb_entities),
                "properties": (df_properties, emb_properties),
            }
            with self.collection.batch.dynamic() as batch:
                for key, (df, vector) in enterprise_df_vectors.items():
                    for i, row in df.iterrows():
                        batch.add_object(
                            properties={**row.to_dict(), "type": key},
                            vector=vector[i].tolist(),
                        )

    def search_entities(self, q: str, k: int = 5) -> pd.DataFrame:
        return self._search(q, type="entities", k=k)

    def search_properties(self, q: str, k: int = 5) -> pd.DataFrame:
        return self._search(q, type="properties", k=k)

    def get_related_candidates(
        self,
        q: str,
        property_candidates: list[str] = [],
        threshold: int = 0.5,
        k: int = 5,
    ) -> dict[str, list[str]]:
        tokens = self._preprocess_into_tokens(q)
        ngrams = self._generate_ngrams(tokens)
        result = {"entities": [], "properties": []}

        def search(ngram, type, threshold=threshold):
            def format_result(x):
                if type == "properties":
                    record = df_res[df_res["short"] == x][["shortDomain", "shortRange"]]
                    record.rename(
                        columns={
                            "shortDomain": "domain",
                            "shortRange": "range",
                        },
                        inplace=True,
                    )
                    record = record.to_dict(orient="records")[0]
                    if record["domain"] is None:
                        del record["domain"]
                    if record["range"] is None:
                        del record["range"]
                    if record:
                        return f"{x}: {record}"
                    return "{x}: No domain and range"
                return x

            df_res = self._search(ngram, type=type, k=k)
            result = (
                df_res[df_res["score"] >= threshold]["short"]
                .apply(format_result)
                .tolist()
            )
            return type, result

        for ngram in ngrams + property_candidates:
            for type in result.keys():
                type, df_res = search(ngram, type)
                if df_res:
                    result[type].extend(df_res)
                    result[type] = list(set(result[type]))

        return result
