import argparse
import json
import os
import time
import logging

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import dspy

from .dataset import build_context_window
from .pipeline import AnnotationPipeline
from .retriever import retrieve
from .signatures import AssignmentClassifier


# -------------------------
# CONFIG
# -------------------------
model = "gpt-5.4" # TODO

POSSIBLE_ASSIGNMENTS = {f"a{i}" for i in range(1, 8)}
REQUIRED_KEYS = {
    "assignment", 
    "assignment_relation", 
    "solution_seeking", 
    "solution_type", 
    "knowledge_seeking", 
    "cognitive_engagement", 
    "mental_model",
    "related_excerpts",
}
USE_ID_FILE = False

# TODO
dataset_path = "in/path/to/dataset.json" 
output_path = "out/path/to/dataset.json" 

now = datetime.now()

total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def add_usage(usage):
    print(usage)
    if usage:
        for k in total_usage:
            total_usage[k] += usage[k]

class LoggingLM(dspy.LM):
    def forward(self, *args, **kwargs):
        filename = now.strftime("prompts/prompts_%Y-%m-%d_%H-%M-%S.log")
        with open(filename, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("ARGS:\n")
            f.write(str(args) + "\n")
            f.write("KWARGS:\n")
            f.write(str(kwargs) + "\n")

        return super().forward(*args, **kwargs)


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    log_filename = now.strftime("automatic_labeling_logs/run_%Y-%m-%d_%H-%M-%S.log")

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger(__name__)


def load_assignment_descriptions(path: Path):
    files = list(path.glob("a*.md"))
    if not files:
        raise FileNotFoundError(f"No assignment files found in {path}")

    return {
        f.stem: f.read_text(encoding="utf-8")
        for f in files
    }


def setup_model():
    api_key = os.getenv("OPENAI_API_KEY")
    
    lm = LoggingLM( # TODO
        model=f"openai/{model}",
        api_key=api_key,
        cache=False,
        temperature=0.0,
    )

    dspy.configure(lm=lm, track_usage=True)
    return dspy.ChainOfThought(AssignmentClassifier)


def safe_assignment(classifier_output, fallback):
    assignment = classifier_output.assignment

    if assignment not in POSSIBLE_ASSIGNMENTS:
        return fallback, True, assignment

    return assignment, False, None


# -------------------------
# LIMITING CHATS
# -------------------------

def load_chat_ids(file_path="./metadata/chat_ids.txt"): # TODO
    file_path = Path(file_path)

    if not file_path.exists():
        return set()

    return set(file_path.read_text(encoding="utf-8").splitlines())


CHAT_IDS_TODO = load_chat_ids()


# -------------------------
# CORE PROCESSING
# -------------------------
def process_turn(turn, assignment, assignment_reasoning,
                 assignment_desc, pipeline, sorted_turns, i):

    relation_context_window = build_context_window(
        sorted_turns,
        i,
        response_turns=0,
        information_to_add=["assignment_relation", "solution_seeking", "solution_type"],
        information_from=model
    )
    
    solution_seeking_context_window = build_context_window(
        sorted_turns,
        i,
        information_to_add=["assignment_relation", "solution_seeking", "solution_type"],
        information_from=model
    )

    cognitive_window = build_context_window(
        sorted_turns,
        i,
        information_from=model
    )

    docs, _ = retrieve(turn["prompt"], assignment=assignment)
    excerpts = "\n\n".join(docs)
    
    lm = dspy.settings.lm
    lm.history = []

    pipeline_result = pipeline(
        assignment_desc,
        excerpts,
        relation_context_window,
        solution_seeking_context_window,
        cognitive_window,
        turn["prompt"]
    )
    
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    for h in lm.history:
        u = h.get("usage", {})
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)

    usage["total_tokens"] = (
        usage["prompt_tokens"] + usage["completion_tokens"]
    )
    
    add_usage(usage)

    return {
        "assignment": {
            "result": assignment,
            "reasoning": assignment_reasoning
        },
        **pipeline_result,
        "related_excerpts": docs,
    }


def is_fully_labeled(llm_label):
    if not isinstance(llm_label, dict):
        return False
    if not llm_label:
        return False
    return REQUIRED_KEYS.issubset(llm_label.keys())


def process_dataset(dataset, assignment_classifier, pipeline, assignment_descriptions, logger):
    total = 0
    start = time.time()

    for user, chats in dataset.items():
        logger.info(f"User: {user}")

        for chat, turns in chats.items():
            if USE_ID_FILE and chat not in CHAT_IDS_TODO: 
                continue
            logger.info(f"Chat: {chat}")

            sorted_turns = sorted(turns, key=lambda x: x["turnId"])
            original_assignment = sorted_turns[0]["topic"]

            full_chat = build_context_window(
                sorted_turns,
                len(sorted_turns),
                prompt_turns=len(sorted_turns),
                prompt_limit=(150, 150),
                response_turns=0,
                information_from=model
            )

            assignment_result = assignment_classifier(
                assignment_descriptions=assignment_descriptions,
                chat=full_chat,
                original_assignment=original_assignment
            )
            usage = assignment_result.get_lm_usage()
            if usage:
                add_usage(usage[f'openai/{model}'])

            assignment, was_fixed, raw = safe_assignment(
                assignment_result,
                original_assignment
            )

            if was_fixed:
                logger.warning(
                    f"Invalid assignment '{raw}' replaced with '{original_assignment}' "
                    f"for user={user}, chat={chat}"
                )

            for i, turn in enumerate(tqdm(sorted_turns, desc=f"{user}/{chat}")):
                if is_fully_labeled(turn.get("labels", {}).get(model)): 
                    continue
                try:
                    result = process_turn(
                        turn,
                        assignment,
                        assignment_result.reasoning,
                        assignment_descriptions[assignment],
                        pipeline,
                        sorted_turns,
                        i
                    )

                    turn.setdefault("labels", {})
                    turn["labels"][model] = result

                    total += 1
                    logger.info(
                        f"Labeled turn {turn['turnId']} | user={user} chat={chat}"
                    )
                    
                    if total % 50 == 0:
                        with open(output_path, "w") as f:
                            json.dump(dataset, f, indent=2)

                except Exception:
                    logger.exception(
                        f"Turn failed user={user}, chat={chat}, turn={turn['turnId']}"
                    )
                    break

    return total, time.time() - start


# -------------------------
# MAIN
# -------------------------
def main():
    logger = setup_logging()

    classifier = setup_model()
    pipeline = AnnotationPipeline()

    with open(dataset_path) as f:
        dataset = json.load(f)

    assignment_descriptions = load_assignment_descriptions(
        Path("summaries_short")
    )

    total, runtime = process_dataset(
        dataset,
        classifier,
        pipeline,
        assignment_descriptions,
        logger
    )

    logger.info("====================")
    logger.info(f"Total labeled: {total}")
    logger.info(f"Runtime (min): {runtime / 60:.2f}")
    logger.info(f"Usage")
    logger.info(f"{total_usage}")
    logger.info("====================")

    os.makedirs("out", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use-id-file",
        action="store_true",
        help="Whether to load and use the chat ID file"
    )

    args = parser.parse_args()
    USE_ID_FILE = args.use_id_file
    main()