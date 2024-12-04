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
        additional_model_kwargs: dict = {},
    ) -> None:
        super().__init__(
            model_name,
            device,
            use_local_model,
            max_new_tokens,
            always_use_generate_sparql,
            print_output,
            additional_model_kwargs,
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
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "Generate a SPARQL query to answer the question: '{input}'"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=few_shots,
        )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """# INSTRUCTIONS
You are an assistant trained to generate Wikidata SPARQL queries. Use the provided context to generate a valid SPARQL query. Based on the given entities and properties context, please generate a valid SPARQL query to answer the question! Return only the uri or literal only. Always generate following the format instruction.

# CONTEXT
- Entity: {resources}
- Ontology candidates: 
{ontology}

# FORMAT INSTRUCTIONS
{format_instructions}""",
                ),
                few_shot_prompt,
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    "Generate a SPARQL query to answer the question: '{input}'",
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
