# FrOG: Framework of Open GraphRAG

![image](https://github.com/user-attachments/assets/d33c13da-0523-482d-b41d-5d5fcafe6f3d)

<details>
<summary>Abstract</summary>

The rise of large language models (LLMs) has advanced information retrieval, but issues like limited knowledge updating, lack of openness, and hallucinations persist. Retrieval-Augmented Generation (RAG) addresses these, though it lacks interpretability due to reliance on vector-based representations. Our research presents a RAG framework using a knowledge graph (KG) as the primary knowledge base, with only open-source components to allow for user customization. The main pipeline consists of entity linking, retrieval using verbalized sentences or SPARQL query generation, and answer generation, integrated with ontology (properties and classes) retrieval via a vector database. The pipeline is tested on Wikidata, DBpedia, and a local domain-specific KG, achieving accuracies of 0.458, 0.517, and 0.805, respectively. An ablation study further reveals that ontology retrieval is the most critical component in providing context to the LLM for generating accurate SPARQL queries.

</details>

## Installation

1. Ensure you have PyTorch installed. (Recommended to have PyTorch installed with GPU support if you want to run a local LLM)
2. Install dependencies:

   ```bash
   pip install -r requirements.txt --no-dependencies
   ```

3. Set up the Weaviate vector database:

   ```bash
   docker compose up -d
   ```

   Alternatively, you can use Weaviate’s cloud service.

4. Create a `.env` file and fill in the following variables:

   ```
   # If you are using Hugging Face API for models
   HF_TOKEN=

   # If you are using Weaviate cloud service
   WEAVIATE_URL=
   WEAVIATE_API_KEY=
   ```

## Example Usage

```python
from rag import WikidataGraphRAG

rag_engine = WikidataGraphRAG(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=1500,
    use_local_model=True,
    use_local_weaviate_client=True,
)

response = rag_engine.chat(
    "How many programming languages are there?",
    verbose=1, use_cot=True, try_threshold=5
)
print(response)
```

## Evaluation

<details>
<summary>Performance</summary>

#### Wikidata

| Configuration    | Jaccard Similarity |
| ---------------- | ------------------ |
| Mistral NeMo 12B | 0.423              |
| LLaMA 3.1 8B     | 0.427              |
| Qwen2.5 Coder 7B | 0.428              |
| **Qwen2.5 7B**   | **0.458**          |

_Table: Experiment Results on the Downsampled Wikidata QALD-9-Plus Dataset_

#### DBpedia

| Configuration    | Jaccard Similarity |
| ---------------- | ------------------ |
| Mistral NeMo 12B | 0.450              |
| LLaMA 3.1 8B     | 0.444              |
| Qwen2.5 Coder 7B | 0.442              |
| **Qwen2.5 7B**   | **0.517**          |

_Table: Experiment Results on the Downsampled DBpedia QALD-9-Plus Dataset_

#### Local KGs

| Configuration                          | Jaccard Similarity |
| -------------------------------------- | ------------------ |
| Mistral NeMo 12B                       | 0.805              |
| LLaMA 3.1 8B                           | 0.778              |
| Qwen2.5 Coder 7B                       | 0.778              |
| Qwen2.5 7B                             | 0.805              |
| **Mistral NeMo 12B w/o Verbalization** | **0.949**          |
| **Qwen2.5 7B w/o Verbalization**       | **0.949**          |

_Table: Experiment Results on the Local Curriculum KG Dataset_

</details>

#### Run Your Own

You can evaluate the framework using the provided `eval.py` script. Below is an overview of the evaluation parameters and their usage.

##### Command-Line Arguments

- `--knowledge-source`: Specify the knowledge source to use. Options: `wikidata`, `dbpedia`, `enterprise`.
- `--model-name`: The name of the model to use.
- `--local-model`: Use the model locally (default: `True`).
- `--local-weaviate-client`: Use the Weaviate client locally (default: `True`).
- `--max-new-tokens`: Maximum number of new tokens (default: `1500`).
- `--test-df-path`: Path to the test dataframe (default: `data/eval/qald-9-downsampled-test-latest.json`).
- `--always-use-generate_sparql`: Always use SPARQL query generation (default: `False`).
- `--use-cot`: Use Chain of Thought reasoning (default: `True`).
- `--llm-try-threshold`: LLM try threshold for retrying queries (default: `10`).
- `--log-file-path`: Path to the log file (default: `logs/eval_log.txt`).
- `--use-transform-factoid`: Enable transform factoid reasoning (default: `False`).

### Example Command

```bash
python eval.py \
    --knowledge-source="wikidata" \
    --model-name="Qwen/Qwen2.5-7B-Instruct" \
    --log-file-path="logs/log.txt" \
    --test-df-path="data/eval/qald-9-downsampled-test-latest.json" \
    --llm-try-threshold=5 \
    --local-model \
    --no-local-weaviate-client \
    --use-cot
```

## Wikidata Integration
For Wikidata, we utilize Wikidata SPARQL Endpoint to execute SPARQL queries and retrieve relevant data. However, to facilitate the vector 
store indexing process, we maintain an offline copy of the list of properties (originally obtained through SPARQL query execution using the Wikidata SPARQL endpoint). For the entity linking stage, FrOG uses the [Wikidata API](https://www.wikidata.org/w/api.php) (specifically its search capabilities) to find KG-specific entities and classes that match entities detected in the user's question. This helps in disambiguating entities and retrieving their URIs.

## Customization

### Enterprise Knowledge Graph Integration

To integrate a custom enterprise knowledge graph:

1. Specify the `turtle_file_path` parameter when initializing `EnterpriseGraphRAG`.
2. Modify the prompt templates in `rag/enterprise.py` and `ENTERPRISE_GENERATE_SPARQL_FEW_SHOTS` few shots in `few_shots.py` to suit your use case.

## Acknowledgments

This framework is built with open-source tools to foster transparency and customization in the research community. Contributions and suggestions are welcome! We also extend our gratitude to Wikimedia Foundation Indonesia for their generous support in funding this research project.
