import pandas as pd
import os
import argparse
from tqdm import tqdm
import warnings

from dbpedia_graph_rag import DBPediaGraphRAG
from few_shots import GENERATE_SPARQL_FEW_SHOTS

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
    model_name: str,
    local=True,
    max_new_tokens: int = 1500,
    test_df_path: str = "data/eval/qald-9-downsampled-test-latest.json",
    always_use_generate_sparql: bool = False,
    llm_try_threshold: int = 10,
    log_file_path: str = "logs/eval_log.txt",
):
    scores = []
    print("Loading test dataframe...")
    test_df = pd.read_json(test_df_path)
    print("Test dataframe loaded.")

    print("Initializing DBPediaGraphRAG...")
    print(f"Local mode: {local}")
    dbpedia_rag = DBPediaGraphRAG(
        model_name=model_name,
        local=local,
        max_new_tokens=max_new_tokens,
        generate_sparql_few_shot_messages=GENERATE_SPARQL_FEW_SHOTS,
        always_use_generate_sparql=always_use_generate_sparql,
    )
    print("DBPediaGraphRAG initialized.")

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "w") as log_file:
        for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
            if i == 3:
                break
            try:
                log_file.write(f"QUESTION {i+1}\n")
                question = row["query_name"]
                true_query = row["cleaned_dbpedia"]
                ground_truth = dbpedia_rag.api.execute_sparql_to_df(true_query)
                generated_factoid_question, generated_query, res = dbpedia_rag.run(
                    question, verbose=0, try_threshold=llm_try_threshold
                )
                pred = pd.DataFrame(res)
                score = compare_two_dataframes(ground_truth, pred)
                log_file.write(
                    f"Question: {question},\nGenerated Factoid Question: {generated_factoid_question},\nTrue Query: {true_query},\n"
                    f"Generated Query: {generated_query if generated_query is not None else 'using verbalization'},\n"
                    f"Top 10 Ground Truth: {ground_truth[:10]}\nTop 10 Generated Answer: {pred[:10]}\nScore: {score}\n"
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
        "--model_name", type=str, required=True, help="Name of the model to use."
    )
    parser.add_argument(
        "--local",
        type=bool,
        default=True,
        help="Whether to run locally.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1500, help="Maximum number of new tokens."
    )
    parser.add_argument(
        "--test_df_path",
        type=str,
        default="data/eval/qald-9-downsampled-test-latest.json",
        help="Path to the test dataframe.",
    )
    parser.add_argument(
        "--always_use_generate_sparql",
        type=bool,
        default=False,
        help="Always use generate SPARQL.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--llm_try_threshold", type=int, default=10, help="LLM try threshold."
    )
    parser.add_argument(
        "--log_file_path",
        type=str,
        default="logs/eval_log.txt",
        help="Path to the log file.",
    )

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        local=args.local,
        max_new_tokens=args.max_new_tokens,
        test_df_path=args.test_df_path,
        always_use_generate_sparql=args.always_use_generate_sparql,
        llm_try_threshold=args.llm_try_threshold,
        log_file_path=args.log_file_path,
    )
