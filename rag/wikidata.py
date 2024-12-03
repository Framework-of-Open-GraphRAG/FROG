import torch
import pandas as pd
from dotenv import load_dotenv

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)


from pydantic import BaseModel, Field
from typing import List, Optional

from few_shots import (
    EXTRACT_ENTITY_FEW_SHOTS,
    WIKIDATA_GENERATE_SPARQL_FEW_SHOTS,
    GENERATE_RELATED_PROPERTIES_FEW_SHOTS,
)
from utils.api import WikidataAPI
from verbalization import WikidataVerbalization
from property_retrieval import WikidataPropertyRetrieval
from .base import BaseGraphRAG

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WikidataGraphRAG(BaseGraphRAG):
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = DEVICE,
        use_local_model: str = True,
        max_new_tokens: int = 1500,
        property_retrieval: Optional[WikidataPropertyRetrieval] = None,
        generate_sparql_few_shot_messages: Optional[List[dict]] = None,
        always_use_generate_sparql: bool = False,
        use_local_weaviate_client: bool = True,
        print_output: bool = False,
    ) -> None:
        super().__init__(
            model_name,
            device,
            use_local_model,
            max_new_tokens,
            always_use_generate_sparql,
            print_output,
        )
        self.api = WikidataAPI()
        self.verbalization = WikidataVerbalization(
            model_name="jinaai/jina-embeddings-v3",
            query_model_encode_kwargs={
                "task": "retrieval.query",
                "prompt_name": "retrieval.query",
            },
            passage_model_encode_kwargs={
                "task": "retrieval.passage",
                "prompt_name": "retrieval.passage",
            },
        )
        if generate_sparql_few_shot_messages is None:
            self.generate_sparql_few_shot_messages = WIKIDATA_GENERATE_SPARQL_FEW_SHOTS
        else:
            self.generate_sparql_few_shot_messages = generate_sparql_few_shot_messages
        if property_retrieval is None:
            df_properties = pd.read_csv("./data/wikidata_ontology/properties.csv")
            self.property_retrieval = WikidataPropertyRetrieval(
                df_properties,
                embedding_model_name="jinaai/jina-embeddings-v3",
                is_local_client=use_local_weaviate_client,
            )
        else:
            self.property_retrieval = property_retrieval

    def extract_entity(self, question, try_threshold=10):
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=EXTRACT_ENTITY_FEW_SHOTS,
        )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """- You are an entity extractor.
- Extract entities from the given question! DO NOT hallucinate and only provide entities that are present in the question.
- These entities usage is to find the most appropriate entity ID from wikidata to be used in SPARQL queries.
- If there is no entity in the question, return empty list.
- Sort the entities based on the importance of the entity in the question.
- ONLY return the entities. DO NOT return anything else.
- DO NOT include adjectives like 'Highest', 'Lowest', 'Biggest', etc in the entity.
- DO NOT provide any extra information, for instance explanation inside a brackets like '(population)', '(area)', '(place)', '(artist)', etc
- DO NOT include any explanations or apologies in your responses.
- Remove all stop words, including conjunctions like 'and' and prepositions like 'in' and 'on' from the extracted entity.
- Make the entity singular, not plural. For instance, if the entity is foods, then transform it into food.
- DO NOT separate Proper Names, e.g. 'Amazon River' should be returned as 'Amazon River'.
- Even if there is only one entity, alwayas return as a list.

Based on the query given, extract all entities from it and return the extracted entities in the format below.
{format_instructions}""",
                ),
                few_shot_prompt,
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        return super().extract_entity(question, chat_prompt_template, try_threshold)

    def get_most_appropriate_entity_uri(
        self,
        entity: str,
        question: str,
        retrieved_entities: list[dict],
        try_threshold: int = 10,
    ) -> str:
        class Entity(BaseModel):
            """
            Represents the most appropriate Wikidata entity ID selected from a given list of retrieved entities.
            This ID is used in Wikidata queries to accurately answer the user's question.
            """

            id: str = Field(
                ...,
                description="Wikidata Entity ID",
            )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """For the entity given, find the most appropriate entity ID from the list of retrieved entities given to be used in Wikidata queries to answer the given question! ONLY return the entity ID from the list of retrieved entities given. DO NOT return anything else and DO NOT hallucinate. DO NOT include any explanations or apologies in your responses.
Based on the entity given, get the most appropriate entity ID from it and return the ID in the format below.
{format_instructions}""",
                ),
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    """Retrieved entities:
{retrieved_entities}

Question:
{question}

Entity: 
{input}

Entity ID:""",
                ),
            ]
        )

        parsed_entity = super().get_most_appropriate_entity_uri(
            entity,
            question,
            retrieved_entities,
            Entity,
            chat_prompt_template,
            try_threshold,
        )

        if parsed_entity is None:
            return None
        return parsed_entity.id

    def generate_related_properties(
        self, question: str, try_threshold: int = 10
    ) -> list[str]:
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=GENERATE_RELATED_PROPERTIES_FEW_SHOTS,
        )

        class RelatedProperty(BaseModel):
            """
            Represents a list of Wikidata property labels relevant to answering a specific question posed by the user.
            These properties are selected based on their appropriateness for extracting the required information
            to provide an accurate and concise answer.
            """

            properties: List[str] = Field(
                ...,
                description="List of Wikidata property label that is appropriate to answer the given question by user.",
            )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Generate 3 Wikidata property label that is appropriate to answer the given question by user. DO NOT include the ID. DO NOT include any explanations or apologies in your responses. No pre-amble.
            
Answer it in the format below. 
{format_instructions}""",
                ),
                few_shot_prompt,
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    "{input}",
                ),
            ]
        )

        related_property = super().generate_related_properties(
            question,
            RelatedProperty,
            chat_prompt_template,
            try_threshold,
        )
        if related_property is None:
            return []
        return related_property.properties

    def generate_sparql(
        self,
        question: str,
        entities: list[str],
        few_shots: list[dict[str, str]],
        use_cot: bool = True,
        verbose: bool = False,
        try_threshold: int = 10,
    ) -> tuple[str, list[dict[str, str]]]:
        # example_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("human", "Generate a SPARQL query to answer the question: '{input}'"),
        #         ("ai", "{output}"),
        #     ]
        # )
        # few_shot_prompt = FewShotChatMessagePromptTemplate(
        #     example_prompt=example_prompt,
        #     examples=few_shots,
        # )
        # print(few_shot_prompt)

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """# INSTRUCTIONS
- Generate SPARQL queries to answer the given question!
- To generate the SPARQL, you can utilize the information from the given Entity IDs. You do not have to use it, but if it can help you to determine the ID of the entity, you can use it.
- You will also be provided with the 100 most used properties with its ID. You are only able to generate SPARQL query from these properties. If it requires property that is not provided, then generate empty query like ```sparql```.
- You can also determine the IDs of the entites that aren't provided with your knowledge.
- Generate the SPARQL with chain of thoughts.
- DO NOT include any apologies in your responses.
- ONLY generate the Thoughts and SPARQL query once! DO NOT try to generate the Question!
- When using a property such as P17 (country), you DO NOT need to verify explicitly whether it is Q6256 entity (country).
- DO NOT use LIMIT, ORDER BY, FILTER in the SPARQL query when not explicitly asked in the question!
- DO NOT aggregation function like COUNT, AVG, etc in the SPARQL query when not asked in the question!
- Always use 'en' language for labels as default unless explicitly asked to use another language.
- Be sure to generate a SPARQL query that is valid and return all the asked information in the question.
- Make the query as simple as possible!
- DO NOT hallucinate the thoughts and query!

# CONTEXT
- Entity: {resources}
- Ontology candidates: 
{ontology}


## EXAMPLES
- Question: Cats
Thoughts:
1. The question asks for information about cats, so I need to identify the relevant entities and properties in Wikidata.
2. First, I need to find items that are classified as cats. In Wikidata, "cat" corresponds to the entity with the identifier Q146.
3. To retrieve items that are instances of cats, I will use the property P31, which stands for "instance of."
4. I should also retrieve the labels of these items in a language the user understands. To do this, I'll utilize the SERVICE wikibase:label to get the label in the user's preferred language. If that language is unavailable, I'll default to a multilingual or English label.
SPARQL Query: ```sparql
SELECT ?item ?itemLabel
WHERE
{{
?item wdt:P31 wd:Q146.
SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en". }} # Helps get the label in your language, if not, then default for all languages, then en language
}}
```

- Question: Picture of Cats
Thoughts:
1. The query is focused on retrieving an image associated with the concept of "cats" in Wikidata.
2. In Wikidata, the item representing "cats" is identified by Q146.
3. The property P18 is used to denote images, so I'll look for the image associated with Q146.
4. The result will return the image linked to the "cats" item.
SPARQL Query: ```sparql
SELECT ?image WHERE {{
  wd:Q146 wdt:P18 ?image. # Get the image (P18) of Cats (Q146)
}}
```

- Question: Cats, with pictures
Thoughts:
1. The question now asks for information about cats, specifically including their pictures.
2. As before, I need to identify items that are classified as cats using the P31 property with the value Q146.
3. In addition to retrieving the item labels, I need to find the property that holds images associated with these items. In Wikidata, the property P18 is used for images.
4. I will add P18 to the query to retrieve the image associated with each cat item.
5. Finally, I'll include the SERVICE wikibase:label to ensure the labels are returned in the appropriate language, defaulting to multilingual or English if necessary.
SPARQL Query: ```sparql
SELECT ?item ?itemLabel ?pic WHERE {{
  ?item wdt:P31 wd:Q146;
    wdt:P18 ?pic.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en". }}
}}
```

- Question: Titles of articles about Ukrainian villages on Romanian Wikipedia
Thoughts:
1. The goal is to find articles about villages in Ukraine that exist on the Romanian Wikipedia.
2. First, I need to identify items classified as villages. In Wikidata, villages are represented by Q532.
3. I will then filter these villages to those located in Ukraine, represented by the country code Q212.
4. I need to check if there is a corresponding article for each village on the Romanian Wikipedia. This is done by filtering for schema:isPartOf with the value <https://ro.wikipedia.org/>.
5. Additionally, I will retrieve the titles of these articles on the Romanian Wikipedia (schema:name as page_titleRO).
6. To provide context, I'll also include the labels of these villages in English (LabelEN) and Ukrainian (LabelUK).
7. Finally, I'll limit the query to return up to 300 results.
SPARQL Query: ```sparql
SELECT DISTINCT ?item ?LabelEN ?LabelUK ?page_titleRO WHERE {{
  # item: is a - village
  ?item wdt:P31 wd:Q532 .
  # item: country - Ukraine
  ?item wdt:P17 wd:Q212 .
  # exists article in item that is ro.wiki
  ?article schema:about ?item ; schema:isPartOf <https://ro.wikipedia.org/> ; schema:name ?page_titleRO .
  # wd labels
  ?item rdfs:label ?LabelEN FILTER (lang(?LabelEN) = "en") .
  ?item rdfs:label ?LabelUK FILTER (lang(?LabelUK) = "uk") .
}}
LIMIT 300
```

- Question: Humans who died on August 25, 2001, on the English Wikipedia, ordered by label
Thoughts:
1. The query requires finding humans who died on a specific date: August 25, 2001.
2. In Wikidata, the date of death is represented by the property P570. I need to identify items where this property matches the specified date.
3. The query also focuses on articles available in English Wikipedia. I'll need to retrieve these articles, ensuring they are from the English Wikipedia by filtering with schema:isPartOf.
4. To sort the results by label, I must consider the proper sorting mechanism. I'll use a regex to clean the labels for sorting purposes, accounting for common prefixes in names (e.g., "von," "de") that might affect alphabetical order.
5. I also need to retrieve the item label and description in the appropriate language using the SERVICE wikibase:label.
6. Finally, the results should be ordered by the cleaned label (?sortname) and the original label.
SPARQL Query: ```sparql
SELECT ?item ?articlename ?itemLabel ?itemDescription ?sl
WHERE {{
VALUES ?dod {{"+2001-08-25"^^xsd:dateTime}}
    ?dod ^wdt:P570 ?item .
    ?item wikibase:sitelinks ?sl .
    ?item ^schema:about ?article .
    ?article schema:isPartOf <https://en.wikipedia.org/>;
    schema:name ?articlename .
SERVICE wikibase:label
    {{
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en" .
    ?item rdfs:label ?itemLabel .
    ?item schema:description ?itemDescription .
    }}
BIND(REPLACE(?itemLabel, "^.*(?<! [Vv][ao]n| [Dd][aeiu]| [Dd][e][lns]| [Ll][ae]) (?!([SJ]r\\.?|[XVI]+)$)", "") AS ?sortname)
}} ORDER BY ASC(UCASE(?sortname)) ASC(UCASE(?itemLabel))
```

- Question: The top 10 heaviest humans
Thoughts:
1. The goal is to identify and list the top 10 heaviest humans based on their recorded weight.
2. Humans are represented in Wikidata by the entity Q5.
3. The property P2067 represents the mass of an individual.
4. To extract the relevant data, I'll search for humans (Q5) who have a recorded mass (P2067).
5. The query should order these individuals by their mass in descending order to find the heaviest.
6. I'll limit the results to the top 10 entries.
7. Additionally, I will include the labels for each individual in multiple languages, prioritizing the user's language settings, and falling back to English, Spanish, French, and German.
SPARQL Query: ```sparql
SELECT ?item ?itemLabel ?mass
WHERE {{
{{
    SELECT ?item ?mass WHERE {{
    ?item wdt:P31 wd:Q5;
            p:P2067/psn:P2067/wikibase:quantityAmount ?mass.
    }}
    ORDER BY DESC(?mass)
    LIMIT 10
}}
SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en,es,fr,de" }}
}}
ORDER BY DESC(?mass)
```

- Question: Number of humans in Wikidata
Thoughts:
1. The question asks for the total number of humans recorded in Wikidata.
2. To find this, I need to identify items that are classified as humans. In Wikidata, the entity for "human" is represented by Q5.
3. I'll use the P31 property, which stands for "instance of," to find all items that are instances of humans.
4. Since the question asks for a count, I'll use the COUNT(*) function to calculate the total number of items that match this criterion.
SPARQL Query: ```sparql
SELECT (COUNT(*) AS ?count)
WHERE {{
?item wdt:P31 wd:Q5 .
}}
```

- Question: List of countries ordered by the number of their cities with a female mayor
Thoughts:
1. The goal is to find countries and list them based on the number of cities within each country that have a female mayor.
2. First, I need to identify instances of cities. In Wikidata, cities or their subclasses are represented by Q515.
3. To find cities with female mayors, I'll use the P6 property, which indicates the head of government. I need to ensure that the head of government is female, which is represented by Q6581072 in Wikidata.
4. I'll also filter out any entries where the mayor's term has ended by checking for the absence of the P582 property (end date).
5. Next, I'll retrieve the country associated with each city using the P17 property.
6. The results should be grouped by country and ordered by the count of cities with a female mayor in descending order.
7. The query will include labels for countries, prioritized by the "ru" (Russian) language, and falling back to "en" (English) if needed.
8. Finally, I'll limit the results to the top 100 countries.
SPARQL Query: ```sparql
SELECT ?country ?countryLabel (count(*) AS ?count)
WHERE
{{
    ?city wdt:P31/wdt:P279* wd:Q515 . # find instances of subclasses of city
    ?city p:P6 ?statement .           # with a P6 (head of goverment) statement
    ?statement ps:P6 ?mayor .         # ... that has the value ?mayor
    ?mayor wdt:P21 wd:Q6581072 .      # ... where the ?mayor has P21 (sex or gender) female
    FILTER NOT EXISTS {{ ?statement pq:P582 ?x }}  # ... but the statement has no P582 (end date) qualifier
    ?city wdt:P17 ?country .          # Also find the country of the city

    # If available, get the "ru" label of the country, use "en" as fallback:
    SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "ru,en" .
    }}
}}
GROUP BY ?country ?countryLabel
ORDER BY DESC(?count)
LIMIT 100
```

- Question: Average number of children per year
Thoughts:
1. The question asks for the average number of children that people have, grouped by their birth year.
2. I'll first identify individuals (humans) in Wikidata, which are represented by Q5.
3. The property P1971 is used to denote the number of children an individual has. I'll retrieve this information for each person.
4. I'll also retrieve each person's birth date using the P569 property and extract the year from the birth date.
5. The results will be filtered to include only those born after 1900 to ensure more recent and relevant data.
6. The query will then group the data by birth year and calculate the average number of children for each year using the AVG function.
7. Finally, I'll return the birth year (year) and the average number of children (count).
SPARQL Query: ```sparql
SELECT  (str(?year) AS ?year) (AVG( ?_number_of_children ) AS ?count) WHERE {{
  ?item wdt:P31 wd:Q5.
  ?item wdt:P1971 ?_number_of_children.
  ?item wdt:P569 ?_date_of_birth.
  BIND( year(?_date_of_birth) as ?year ).
  FILTER( ?year > 1900)
}}

GROUP BY ?year
```""",
                ),
                # few_shot_prompt,
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    "Question: '{input}'",
                ),
            ]
        )

        return super().generate_sparql(
            question,
            entities,
            chat_prompt_template,
            use_cot,
            verbose,
            try_threshold,
        )

    def process_context(
        self, question: str, context: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]]]:
        if type(context) == list:
            if len(context) > 0:
                if list(context[0].values())[0].startswith("http://www.wikidata.org/"):
                    context_entities = []
                    for c in context[:50]:
                        context_entities.append(
                            "wd:" + list(c.values())[0].split("/")[-1]
                        )
                    get_label_query = f"""SELECT ?itemLabel WHERE {{
  VALUES ?item {{ {" ".join(context_entities)} }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}"""
                    context, _ = self.api.execute_sparql(get_label_query)
                context_str = f'The answer of "{question}" is '
                for c in context[:50]:
                    for k, v in c.items():
                        context_str += k + " = " + v + ", "
                context_str = context_str[:-2] + "."
            else:
                context_str = "I don't know"
        else:
            context_str = f'The answer of "{question}" is {context}'
        return context_str, context

    def run(
        self,
        question: str,
        use_cot: bool = True,
        output_uri: bool = False,
        verbose: int = 0,
        try_threshold: int = 10,
    ):
        return super().run(
            question,
            use_cot=use_cot,
            output_uri=output_uri,
            use_transform_factoid=False,
            verbose=verbose,
            try_threshold=try_threshold,
        )
