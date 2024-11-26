from utils.api import WikidataAPI
from verbalization.base import BaseVerbalization


class WikidataVerbalization(BaseVerbalization):
    SENTENCE_TEMPLATE = "{s}'s {p} is {o}"
    MANUAL_MAPPING_DICT = {"_": " "}
    PO_TEMPLATE = """
SELECT distinct ?p ?o ?sLabel ?propLabel ?oLabel
WHERE {{
  BIND(wd:{entity} AS ?s) .
  
  ?s ?p ?o .
  FILTER(?p != wd:P18)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
  ?prop wikibase:directClaim ?p .
}}
"""
    SP_TEMPLATE = """
SELECT ?s ?p ?sLabel ?propLabel ?oLabel
WHERE {{
  BIND(wd:{entity} AS ?o) .
  
  ?s ?p ?o .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
  ?prop wikibase:directClaim ?p .
}}
"""

    def __init__(
        self,
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={"trust_remote_code": True},
        query_model_encode_kwargs={},
        passage_model_encode_kwargs={},
    ) -> None:
        super().__init__(
            model_name,
            model_kwargs,
            query_model_encode_kwargs,
            passage_model_encode_kwargs,
        )
        self.api = WikidataAPI()
