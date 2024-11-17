import torch, os
import pandas as pd
from IPython.display import HTML, display
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from xml.sax.saxutils import escape

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.output_parsers import (
    ResponseSchema,
    StructuredOutputParser,
    PydanticOutputParser,
)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.output_parsers.prompts import NAIVE_FIX_PROMPT


from pydantic import BaseModel, Field
from typing import List, Optional

from few_shots import (
    EXTRACT_ENTITY_FEW_SHOTS,
    GENERATE_SPARQL_FEW_SHOTS,
    INTENT_CLASSIFICATION_FEW_SHOTS,
    GENERATE_RELATED_PROPERTIES_FEW_SHOTS,
)
from utils.api import DBPediaAPI
from utils.verbalization import Verbalization
from utils.property_retrieval import DBPediaPropertyRetrieval
from utils.helper import separate_camel_case

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DBPediaGraphRAG:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = DEVICE,
        local: str = True,
        max_new_tokens: int = 1500,
        property_retrieval: Optional[DBPediaPropertyRetrieval] = None,
        generate_sparql_few_shot_messages: Optional[List[dict]] = None,
        always_use_generate_sparql: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local = local
        self.api = DBPediaAPI()
        self.verbalization = Verbalization(model_name="multi-qa-mpnet-base-cos-v1")
        if generate_sparql_few_shot_messages is None:
            self.generate_sparql_few_shot_messages = GENERATE_SPARQL_FEW_SHOTS
        else:
            self.generate_sparql_few_shot_messages = generate_sparql_few_shot_messages
        df_classes = pd.read_csv("./data/dbpedia_ontology/classes.csv")
        df_oproperties = pd.read_csv("./data/dbpedia_ontology/oproperties.csv")
        df_dproperties = pd.read_csv("./data/dbpedia_ontology/dproperties.csv")
        if property_retrieval is None:
            self.property_retrieval = DBPediaPropertyRetrieval(
                df_classes,
                df_oproperties,
                df_dproperties,
            )
        else:
            self.property_retrieval = property_retrieval
        model_kwargs = {
            "temperature": 0,
            "device": self.device,
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
        }
        if self.local:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=HF_TOKEN
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, token=HF_TOKEN
            )
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **model_kwargs,
            )
            llm = HuggingFacePipeline(pipeline=pipe)
        else:
            self.tokenizer = None
            self.model = None
            llm = HuggingFaceEndpoint(
                repo_id=self.model_name,
                **model_kwargs,
                huggingfacehub_api_token=HF_TOKEN,
            )
        self.chat_model = ChatHuggingFace(llm=llm)
        self.always_use_generate_sparql = always_use_generate_sparql

    def contains_multiple_entities(self, question: str) -> bool:
        keywords = [
            "and",
            "or",
            "nor",
            "as well as",
            "both",
            "along with",
            "together with",
            "in addition to",
            "besides",
            "also",
        ]
        question = question.lower()
        return any(
            f" {keyword} " in question
            or question.startswith(f"{keyword} ")
            or question.endswith(f" {keyword}")
            for keyword in keywords
        )

    def get_propertty_domain_range(self, property_uri: str) -> dict[str, str]:
        query = f"""SELECT ?domain ?range
WHERE {{
    {property_uri} rdfs:domain ?domain ;
                   rdfs:range ?range .
}}
"""
        return self.api.execute_sparql_to_df(query).drop_duplicates().to_dict("records")

    def handle_parsing_error(
        self,
        llm_chain: LLMChain,
        output_parser,
        messages: list[dict],
        question: str,
        chat_history_placeholder: str = "chat_history",
        try_threshold: int = 10,
    ) -> tuple[any, list[str]]:
        fix_llm_chain = NAIVE_FIX_PROMPT | self.chat_model | StrOutputParser()
        completion = llm_chain.invoke(
            {chat_history_placeholder: messages, "input": question}
        )
        messages.append(HumanMessage(content=question))

        completion_parsed = None
        while try_threshold > 0:
            try:
                completion_parsed = output_parser.parse(completion)
                break
            except Exception as e:
                display(
                    HTML(f"""<code style='color: red;'>{(escape(str(e)))}</code>""")
                )
                try_threshold -= 1

                try:
                    completion = fix_llm_chain.invoke(
                        {
                            "instructions": output_parser.get_format_instructions(),
                            "completion": completion,
                            "error": repr(e),
                        }
                    )
                except Exception as e:
                    display(
                        HTML(f"""<code style='color: red;'>{(escape(str(e)))}</code>""")
                    )
                    break

        messages.append(completion)
        if completion_parsed is not None:
            return completion_parsed, messages
        return None, messages

    def transform_to_factoid_question(
        self, question: str, try_threshold: int = 10
    ) -> str:
        response_schemas = [
            ResponseSchema(
                name="question",
                description="factoid question transformed from user's question or instruction",
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        query_reformulation_chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are transforming user query into a factoid question or instruction. A factoid question is a type of question that seeks a brief, factual answer, typically related to a specific piece of information such as a date, name, or location. The answers are usually objective and can be verified, often being short and to the point. For example:
- "Who is the president of the United States?"
- "When did World War II end?"
- "What is the capital of France?"

These questions don't require long explanations or complex reasoning and are often used in information retrieval tasks and natural language processing systems to extract concise factual information from large datasets.
Given an input query, convert it to a a factoid question. No pre-amble.

Based on the query given, transform from it and return the factoid question in the format below.
{format_instructions}""",
                ),
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    "{input}",
                ),
            ]
        )
        query_reformulation_chat_prompt = query_reformulation_chat_prompt.partial(
            format_instructions=format_instructions
        )

        llm_chain = (
            query_reformulation_chat_prompt | self.chat_model | StrOutputParser()
        )
        messages = []
        response, messages = self.handle_parsing_error(
            llm_chain, output_parser, messages, question, try_threshold=try_threshold
        )
        if response is None:
            return question
        return response["question"]

    def classify_intent_is_global(self, question: str, try_threshold: int = 10) -> bool:
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=INTENT_CLASSIFICATION_FEW_SHOTS,
        )

        class Intent(BaseModel):
            """Identifying information about entities."""

            is_global: bool = Field(
                ...,
                description="Whether the query is global or local.",
            )

        output_parser = PydanticOutputParser(pydantic_object=Intent)
        format_instructions = output_parser.get_format_instructions()

        query_intent_chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Your task is to classify user queries as either global or local.
- A **global** query asks for general or broad information, often requiring summarization of an entire dataset. For example, "What are the main themes in the dataset?"
- A **local** query asks for specific information from a particular part of the data. For example, "What is the capital of France?"
Based on the query given, decide if it is global or local and return the classification in the format below.
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
        query_intent_chat_prompt = query_intent_chat_prompt.partial(
            format_instructions=format_instructions
        )

        llm_chain = query_intent_chat_prompt | self.chat_model | StrOutputParser()
        messages = []
        intent, messages = self.handle_parsing_error(
            llm_chain, output_parser, messages, question, try_threshold=try_threshold
        )
        if intent is None:
            return True
        return intent.is_global

    def get_most_appropriate_resource_uri(
        self, entity: str, retrieved_resources: list[dict], try_threshold: int = 10
    ) -> str:
        class Resource(BaseModel):
            """Identifying information about entities."""

            uri: str = Field(
                ...,
                uri="Resource URI",
            )

        output_parser = PydanticOutputParser(pydantic_object=Resource)
        format_instructions = output_parser.get_format_instructions()

        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """For the entity given, find the most appropriate resource uri from the list of retrieved resources given to be used in DBPedia queries to answer the given question! ONLY return the resource uri from the list of retrieved resources given. DO NOT return anything else and DO NOT hallucinate. DO NOT include any explanations or apologies in your responses.
Based on the entity given, get the most appropriate resource uri from it and return the uri in the format below.
{format_instructions}""",
                ),
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    """Retrieved resources:
{retrieved_resources}

Entity: 
{input}

Resource URI:""",
                ),
            ]
        )
        final_prompt = final_prompt.partial(
            format_instructions=format_instructions,
            retrieved_resources=retrieved_resources,
        )
        llm_chain = final_prompt | self.chat_model | StrOutputParser()
        messages = []
        resource, messages = self.handle_parsing_error(
            llm_chain, output_parser, messages, entity, try_threshold=try_threshold
        )
        if resource is None:
            return None
        return resource.uri.replace("dbr:", "http://dbpedia.org/resource/")

    def extract_entity(self, question: str, try_threshold: int = 10) -> list[str]:
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

        class Entities(BaseModel):
            """Identifying information about entities."""

            names: List[str] = Field(
                ...,
                description="All the entities appearing in the text, sorted by importance.",
            )

        output_parser = PydanticOutputParser(pydantic_object=Entities)
        format_instructions = output_parser.get_format_instructions()

        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """- You are an entity extractor.
- Extract the entities from the given question! DO NOT hallucinate and only provide entities that are present in the question.
- These entities usage is to find the most appropriate entity URI/ID from dbpedia/wikidata to be used in SPARQL queries.
- If there is no entity in the question, return empty list.
- Sort the entities based on the importance of the entity in the question.
- ONLY return the entities. DO NOT return anything else.
- DO NOT include adjectives like 'Highest', 'Lowest', 'Biggest', etc in the entity.
- DO NOT provide any extra information, for instance explanation inside a brackets like '(population)', '(area)', '(place)', '(artist)', etc
- DO NOT include any explanations or apologies in your responses.
- Remove all stop words, including conjunctions like 'and' and prepositions like 'in' and 'on' from the extracted entity.
- Make the entity singular, not plural. For instance, if the entity is foods, then transform it into food.
- Even if there is only one entity, alwayas return as a list.

Based on the query given, extract the entities from it and return the extracted entities in the format below.
{format_instructions}""",
                ),
                few_shot_prompt,
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        final_prompt = final_prompt.partial(format_instructions=format_instructions)
        llm_chain = final_prompt | self.chat_model | StrOutputParser()
        messages = []
        entities, messages = self.handle_parsing_error(
            llm_chain, output_parser, messages, question, try_threshold=try_threshold
        )
        if entities is None:
            return []
        return entities.names

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
            """Identifying information about entities."""

            properties: List[str] = Field(
                ...,
                description="List of DBPedia property that is appropriate to answer the given question by user. Max 3.",
            )

        output_parser = PydanticOutputParser(pydantic_object=RelatedProperty)
        format_instructions = output_parser.get_format_instructions()

        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Generate list of DBPedia property that is appropriate to answer the given question by user. DO NOT include any explanations or apologies in your responses. No pre-amble.
            
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

        final_prompt = final_prompt.partial(
            format_instructions=format_instructions,
        )

        llm_chain = final_prompt | self.chat_model | StrOutputParser()
        messages = []

        related_property, messages = self.handle_parsing_error(
            llm_chain,
            output_parser,
            messages,
            question,
            try_threshold=try_threshold,
        )
        if related_property is None:
            return []
        return related_property.properties

    def _parse_property_context_string(self, ontology: dict[str, list[str]]) -> str:
        def parallel_search(key, name):
            domain_range = self.get_propertty_domain_range(name)
            return key, name, domain_range

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(parallel_search, key, c)
                for key, cand in ontology.items()
                if cand
                for c in cand
                if key != "classes"
            ]

            result = {}
            for future in futures:
                key, name, domain_range = future.result()
                result.setdefault(key, []).append(
                    {"name": name, "domain_range": domain_range}
                )

        properties_context = ""
        if ontology["classes"]:
            properties_context += f"    - classes: {ontology['classes']}\n"
        for key, value in result.items():
            properties_context += f"    - {key}: \n"
            for prop in value:
                name = prop["name"]
                domain_range = prop["domain_range"]
                if domain_range:
                    properties_context += f"        - {name}: {domain_range}\n"
                else:
                    properties_context += f"        - {name}: No domain and range\n"

        return properties_context

    def get_dbpedia_results(
        self,
        question: str,
        entities: list[str],
        verbose: bool = False,
        try_threshold: int = 10,
    ) -> tuple[str, list[dict[str, str]]]:
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.generate_sparql_few_shot_messages,
        )

        class SPARQLQueryResults(BaseModel):
            """Identifying information about entities."""

            thoughts: List[str] = Field(
                ...,
                description="Thoughts to generate SPARQL query to answer the user's question.",
            )
            sparql: str = Field(
                ...,
                description="SPARQL query to answer the user's question.",
            )

        output_parser = PydanticOutputParser(pydantic_object=SPARQLQueryResults)
        format_instructions = output_parser.get_format_instructions()

        resources = ""
        for entity in entities:
            resources += f"    - All possible resources URIs for {entity} are "
            resources += str(self.api.get_entities(entity, k=3)[0])
            resources += "\n"
        if verbose:
            resources_tmp = escape(resources).replace("\n", "<br/>")
            display(
                HTML(
                    f"""<code style='color: green;'>Retrieved Resources: <br/>{resources_tmp}</code>"""
                )
            )

        related_properties = self.generate_related_properties(question)[:3]
        for i in range(len(related_properties)):
            related_properties[i] = separate_camel_case(related_properties[i]).lower()
        if verbose:
            display(
                HTML(
                    f"""<code style='color: green;'>Generated Related Properties: {escape(str(related_properties))}</code>"""
                )
            )

        ontology = self.property_retrieval.get_related_candidates(
            question, property_candidates=related_properties, threshold=0.5
        )
        properties_context = self._parse_property_context_string(ontology)
        if verbose:
            properties_context_tmp = escape(properties_context).replace("\n", "<br/>")
            display(
                HTML(
                    f"""<code style='color: green;'>Retrieved Ontology: <br/>{properties_context_tmp}</code>"""
                )
            )

        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a DBPedia SPARQL generator.
- Based on the context given, generate SPARQL query for DBPedia that would answer the user's question!
- You will also be provided with the resources with its URI, ontology candidates consisting of classes, object properties, and data properties. You are only able to generate SPARQL query from the given context. Please determine to use the most appropriate one.
- To generate the SPARQL, you can utilize the information from the given Entity URIs. You do not have to use it, but if it can help you to determine the URI of the entity, you can use it.
- USE the URI from resources given if you need to query more specific entity. On the other hand, USE classes from ontology if it's more general.
- Generate the SPARQL with chain of thoughts.
- DO NOT include any apologies in your responses.
- ONLY generate the Thoughts and SPARQL query once! DO NOT try to generate the Question!
- DO NOT use LIMIT, ORDER BY, FILTER in the SPARQL query when not explicitly asked in the question!
- DO NOT aggregation function like COUNT, AVG, etc in the SPARQL query when not asked in the question!
- Be sure to generate a SPARQL query that is valid and return all the asked information in the question.
- Make the query as simple as possible!
- DO NOT hallucinate the thoughts and query!
- Always use english ('en') language for labels in all columns as default unless explicitly asked to use another language.
- Once again, always use english as the label language in all columns if not explicitly asked to use another language.
Context:
- Resources retrieved:
{resources}
- Ontology candidates retrieved:
{ontology}
Based on the query and context given, generate the thoughts and SPARQL query from it and return the thoughts and SPARQL query in the format below.
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

        final_prompt = final_prompt.partial(
            resources=resources,
            ontology=properties_context,
            format_instructions=format_instructions,
        )
        llm_chain = final_prompt | self.chat_model | StrOutputParser()
        messages = []

        curr_question = question
        while True:
            sparql_query_result, messages = self.handle_parsing_error(
                llm_chain,
                output_parser,
                messages,
                curr_question,
                try_threshold=try_threshold,
            )
            if (
                sparql_query_result is None
                or sparql_query_result.sparql == ""
                or sparql_query_result.sparql is None
            ):
                display(
                    HTML(
                        f"""<code style='color: green;'>Sorry, we are not supported with this kind of query yet.</code>"""
                    )
                )
                return None, []
            if verbose:
                thoughts_tmp = escape(str(sparql_query_result.thoughts))
                sparql_tmp = escape(sparql_query_result.sparql).replace("\n", "<br/>")
                display(HTML(f"""<code style='color: green;'>{thoughts_tmp}</code>"""))
                display(HTML(f"""<code style='color: green;'>{sparql_tmp}</code>"""))
            try:
                result, err = self.api.execute_sparql(sparql_query_result.sparql)
            except Exception as e:
                display(HTML(f"""<code style='color: red;'>{e}</code>"""))
                result, err = [], str(e)

            if len(result) == 0 and try_threshold > 0:
                # failed
                try_threshold -= 1
                if verbose:
                    display(
                        HTML(f"""<code style='color: green;'>Trying again...</code>""")
                    )

                curr_question = f"""The SPARQL query you generated above to answer '{question}' is wrong {f"and it produces this error: '{err}'" if err is not None else "because it produces empty results"}, please fix the query and generate SPARQL again! You may try to use another property or restucture the query.
DO NOT include any explanations or apologies in your responses. No pre-amble. Make sure to still answer using chain of thoughts and structure based on the format instruction defined in system prompt."""
            else:
                # success
                break
        return sparql_query_result.sparql, result

    def run(
        self,
        question: str,
        verbose: int = 0,
        try_threshold: int = 10,
    ) -> tuple[str, str, list[dict[str, str]]]:
        factoid_question = self.transform_to_factoid_question(question)
        if verbose == 1:
            display(
                HTML(
                    f"""<code style='color: green;'>Factoid Question: {escape(factoid_question)}</code>"""
                )
            )

        extracted_entities = self.extract_entity(factoid_question)
        if verbose == 1:
            display(
                HTML(
                    f"""<code style='color: green;'>Entities: {escape(str(extracted_entities))}</code>"""
                )
            )

        if self.always_use_generate_sparql:
            intent_is_global = True
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Intent is always global because always_use_generate_sparql is set to True</code>"""
                    )
                )
        else:
            intent_is_global = self.classify_intent_is_global(question)
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Intent is global: {escape(str(intent_is_global))}</code>"""
                    )
                )

        if not intent_is_global and self.contains_multiple_entities(question):
            intent_is_global = True
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Use SPARQL generation because the question contains multiple entities.</code>"""
                    )
                )

        query = None
        if intent_is_global:
            query, result = self.get_dbpedia_results(
                factoid_question,
                extracted_entities,
                verbose=verbose > 0,
                try_threshold=try_threshold,
            )
        else:
            entity = extracted_entities[0]
            retrieved_resources = self.api.get_entities(entity, k=3)[0]
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Retrieved Resources: {escape(str(retrieved_resources))}</code>"""
                    )
                )
            entity_uri = self.get_most_appropriate_resource_uri(
                entity, retrieved_resources
            )
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Entity URI: {escape(entity_uri)}</code>"""
                    )
                )

            is_error = False
            try:
                result, similarities = self.verbalization.run(
                    factoid_question, entity_uri
                )
                if verbose == 1:
                    display(
                        HTML(
                            f"""<code style='color: green;'>Result: {escape(str(result))}<br/>Similarities: {escape(str(similarities))}</code>"""
                        )
                    )
            except Exception as e:
                is_error = True
                if verbose == 1:
                    display(
                        HTML(f"""<code style='color: red;'>{escape(str(e))}</code>""")
                    )
            if is_error or similarities < 0.65 or len(result) == 0:
                query, result = self.get_dbpedia_results(
                    factoid_question,
                    extracted_entities,
                    verbose=verbose > 0,
                    try_threshold=try_threshold,
                )
        return factoid_question, query, result

    def chat(self, question: str, verbose: int = 0, try_threshold: int = 10) -> str:
        factoid_question, _, dbpedia_context = self.run(
            question, verbose=verbose, try_threshold=try_threshold
        )
        if verbose == 1:
            display(
                HTML(
                    f"""<code style='color: green;'>{escape(str(dbpedia_context))}</code>"""
                )
            )

        final_prompt = ChatPromptTemplate.from_messages(
            [
                #                 (
                #                     "system",
                #                     """- You are an assistant for question-answering tasks.
                # - Use the following pieces of retrieved context to answer the question.
                # - If you don't know the answer, just say that you don't know.
                # - Try your best to use the given context as the answer!
                # - DO NOT hallucinate and only provide answers from the given context.
                # - DO NOT make up an answer.
                # - If the context does not explicitly say related to the question, just ASSUME that it is the answer of the question.
                # - Answer the question in a natural way like you are the one who know the context, DO NOT mention like "according to the context", etc.
                # - Answer it using complete sentences!
                # - If the question is about retrieving information that is limited to a certain amount, make sure to return all the results from the context that match the limited amount.
                # - If the answer is a list of items, then you should answer in bullet points!
                # - When no context is provided, just answer "I don't know".""",
                #                 ),
                (
                    "human",
                    #                     """Answer the following question strictly using the context provided below. Assume that if context is provided, it is the correct answer. If the information to answer the question is not available in the context, respond with "I don't know."
                    # Context:
                    # {dbpedia_context}
                    # Question: {input}
                    # Answer:""",
                    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {input} 
Context: {dbpedia_context} 
Answer:""",
                ),
            ]
        )
        if type(dbpedia_context) == list:
            if len(dbpedia_context) > 0:
                context_str = f'The answer of "{factoid_question}" is '
                for c in dbpedia_context[:50]:
                    for k in c.keys():
                        context_str += k + " = " + c[k] + ", "
                context_str = context_str[:-2] + "."
            else:
                context_str = "I don't know"
        else:
            context_str = f'The answer of "{factoid_question}" is {dbpedia_context}'
        final_prompt = final_prompt.partial(dbpedia_context=context_str)

        llm_chain = final_prompt | self.chat_model | StrOutputParser()

        response = llm_chain.invoke({"input": factoid_question})
        return response
