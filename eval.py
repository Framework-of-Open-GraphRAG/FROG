import pandas as pd
import os
import argparse
from tqdm import tqdm
import warnings

from rag import WikidataGraphRAG, DBPediaGraphRAG
from few_shots import (
    WIKIDATA_GENERATE_SPARQL_FEW_SHOTS,
    DBPEDIA_GENERATE_SPARQL_FEW_SHOTS,
)

warnings.filterwarnings("ignore")


def compare_two_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    if len(df1.columns) != len(df2.columns):
        return 0

    set1, set2 = set(), set()
    for _, row in df1.iterrows():
        row = list(row)
        row = sorted(row)
        row = tuple(row)
        set1.add(row)

    for _, row in df2.iterrows():
        row = list(row)
        row = sorted(row)
        row = tuple(row)
        set2.add(row)

    return len(set1 & set2) / len(set1 | set2)


def main(
    knowledge_source: str,
    model_name: str,
    use_local_model=True,
    use_local_weaviate_client=True,
    max_new_tokens: int = 1500,
    test_df_path: str = "data/eval/qald-9-downsampled-test-latest.json",
    always_use_generate_sparql: bool = False,
    use_cot: bool = True,
    llm_try_threshold: int = 10,
    log_file_path: str = "logs/eval_log.txt",
):
    scores = []
    print("Loading test dataframe...")
    test_df = pd.read_json(test_df_path)
    print("Test dataframe loaded.")

    print(f"Local mode: {use_local_model}")
    if knowledge_source == "wikidata":
        print("Initializing WikidataGraphRAG...")
        rag_engine = WikidataGraphRAG(
            model_name=model_name,
            use_local_model=use_local_model,
            max_new_tokens=max_new_tokens,
            generate_sparql_few_shot_messages=WIKIDATA_GENERATE_SPARQL_FEW_SHOTS,
            always_use_generate_sparql=always_use_generate_sparql,
            use_local_weaviate_client=use_local_weaviate_client,
        )
        print("WikidataGraphRAG initialized.")
    elif knowledge_source == "dbpedia":
        print("Initializing DBPediaGraphRAG...")
        rag_engine = DBPediaGraphRAG(
            model_name=model_name,
            use_local_model=use_local_model,
            max_new_tokens=max_new_tokens,
            generate_sparql_few_shot_messages=DBPEDIA_GENERATE_SPARQL_FEW_SHOTS,
            always_use_generate_sparql=always_use_generate_sparql,
            use_local_weaviate_client=use_local_weaviate_client,
        )
        print("DBPediaGraphRAG initialized.")
    else:
        raise ValueError("Invalid knowledge source.")

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "w") as log_file:
        for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
            try:
                log_file.write(f"QUESTION {i+1}\n")
                question = row["query_name"]
                true_query = row[f"cleaned_{knowledge_source}"]
                ground_truth = rag_engine.api.execute_sparql_to_df(true_query)
                generated_factoid_question, generated_query, res = rag_engine.run(
                    question,
                    use_cot=use_cot,
                    output_uri=True,
                    verbose=0,
                    try_threshold=llm_try_threshold,
                )
                pred = pd.DataFrame(res)
                score = compare_two_dataframes(ground_truth, pred)
                log_file.write(
                    f"Question: {question},\nGenerated Factoid Question: {generated_factoid_question},\nTrue Query: {true_query},\n"
                    f"Generated Query: {generated_query if generated_query != '' else 'using verbalization'},\n"
                    f"Top 10 Ground Truth:\n{ground_truth[:10]}\nTop 10 Generated Answer:\n{pred[:10]}\nScore: {score}\n"
                )
            except Exception as e:
                log_file.write("ERROR: " + str(e) + "\n")
                score = 0
            scores.append(score)
            current_avg_score = sum(scores) / len(scores)
            log_file.write(f"Current Average Score: {current_avg_score}\n")
            log_file.write(
                "----------------------------------------------------------------------------------------\n"
            )
            log_file.flush()  # Ensure the log is saved after each iteration
        log_file.write("Average Score: " + str(current_avg_score) + "\n")
    print(f"Average Score: {current_avg_score}")
    print("Evaluation done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation script.")
    parser.add_argument(
        "--knowledge-source",
        type=str,
        required=True,
        help="Knowledge source to use.",
        choices=["wikidata", "dbpedia"],
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the model to use."
    )
    parser.add_argument(
        "--local-model",
        type=bool,
        default=True,
        help="Whether to run model locally.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--local-weaviate-client",
        type=bool,
        default=True,
        help="Whether to use weaviate client locally.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1500, help="Maximum number of new tokens."
    )
    parser.add_argument(
        "--test-df-path",
        type=str,
        default="data/eval/qald-9-downsampled-test-latest.json",
        help="Path to the test dataframe.",
    )
    parser.add_argument(
        "--always-use-generate_sparql",
        type=bool,
        default=False,
        help="Always use generate SPARQL.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--use-cot",
        type=bool,
        default=True,
        help="Whether to use Chain of Thought.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--llm-try-threshold", type=int, default=10, help="LLM try threshold."
    )
    parser.add_argument(
        "--log-file-path",
        type=str,
        default="logs/eval_log.txt",
        help="Path to the log file.",
    )

    args = parser.parse_args()

    main(
        knowledge_source=args.knowledge_source,
        model_name=args.model_name,
        use_local_model=args.local_model,
        use_local_weaviate_client=args.local_weaviate_client,
        max_new_tokens=args.max_new_tokens,
        test_df_path=args.test_df_path,
        always_use_generate_sparql=args.always_use_generate_sparql,
        use_cot=args.use_cot,
        llm_try_threshold=args.llm_try_threshold,
        log_file_path=args.log_file_path,
    )
