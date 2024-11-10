import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import xmltodict
import urllib.parse


class BaseAPI:
    def __init__(self, url: str, agent: str) -> None:
        self.sparqlwd = SPARQLWrapper(url, agent=agent)

    def execute_sparql(self, q: str) -> tuple[list[dict], Exception]:
        self.sparqlwd.setQuery(q)
        self.sparqlwd.setReturnFormat(JSON)
        try:
            results = self.sparqlwd.query().convert()
            results_cleaned = []
            for result in results["results"]["bindings"]:
                tmp = dict()
                for header in results["head"]["vars"]:
                    tmp[header] = result[header]["value"]
                results_cleaned.append(tmp)
            return results_cleaned, None
        except Exception as e:
            return [], e

    def execute_sparql_to_df(self, q: str):
        self.sparqlwd.setQuery(q)
        self.sparqlwd.setReturnFormat(JSON)
        results = self.sparqlwd.query().convert()
        df = []
        for result in results["results"]["bindings"]:
            row = {}
            for key, value in result.items():
                row[key] = value["value"]
            df.append(row)

        return pd.DataFrame(df)

    def get_entities(self, entity: str, k: int = 5, lang=None):
        raise NotImplementedError("This method should be overridden by subclasses")


class DBPediaAPI(BaseAPI):
    def __init__(self, url="http://dbpedia.org/sparql") -> None:
        super().__init__(
            url,
            agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
        )

    def get_entities(
        self, entity: str, k: int = 5, lang=None
    ) -> tuple[list[dict[str, str]], Exception]:
        dbpedia_api = "https://lookup.dbpedia.org/api/search?query="
        encoded_param = urllib.parse.quote_plus(entity)

        try:
            data = requests.get(f"{dbpedia_api}/{encoded_param}")
        except Exception as e:
            return [], e

        dict_data = xmltodict.parse(data.content)
        dbr_uri = "http://dbpedia.org/resource/"
        parsed_data = [
            {
                "uri": item["URI"].replace(dbr_uri, "dbr:"),
                "label": item["Label"],
                "description": item.get("Description", ""),
            }
            for item in dict_data["ArrayOfResults"]["Result"][:k]
            if item["URI"].startswith(dbr_uri)
        ]
        return parsed_data, None


class WikidataAPI(BaseAPI):
    def __init__(self, url="https://query.wikidata.org/sparql") -> None:
        super().__init__(
            url,
            agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
        )

    def get_entities(
        self, entity: str, k: int = 5, lang: str = "en"
    ) -> tuple[str, Exception]:
        wikidata_api = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "search": entity,
            "language": lang,
        }

        try:
            data = requests.get(wikidata_api, params=params)
        except Exception as e:
            return [], e

        json_data = data.json()
        parsed_data = [
            {
                "uri": item["id"],
                "label": item["label"],
                "description": item.get("description", ""),
            }
            for item in json_data["search"][:k]
        ]
        return parsed_data, None
