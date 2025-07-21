import logging
import sys
import argparse
from scripts.simulate_events import simulate_events
from ML.build_training_dataset import build_training_dataset
from scripts.s3_to_dynamodb import s3_to_dynamodb
from ML.item_embeddings import generate_item_embeddings
from ML.train_faiss_index import train_faiss_index
from scripts.prepare_evaluation_data import prepare_evaluation_data
from scripts.offline_evaluation import run_offline_evaluation


# Mapping of pipeline steps to functions
PIPELINE_STEPS = {
    "simulate_events": simulate_events,
    "s3_to_dynamodb": s3_to_dynamodb,
    "build_training_dataset": build_training_dataset,
    "generate_item_embeddings": generate_item_embeddings,
    "train_faiss_index": train_faiss_index
}

EVAL_STEPS = {
    "prepare_evaluation_data": prepare_evaluation_data,
    "offline_evaluation": run_offline_evaluation
}

ALL_STEPS = list(PIPELINE_STEPS.keys())
ALL_EVAL = list(EVAL_STEPS.keys())

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    logging.info("Starting AI Recommendation System Pipeline")
    parser = argparse.ArgumentParser(description="Run AI Recommendation System Pipeline")
    parser.add_argument(
        "step",
        choices=["all", "eval"] + ALL_STEPS + ALL_EVAL,
        help="Pipeline step to run. Use 'all' to run full pipeline or 'eval' to run evaluation suite."
    )
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop the pipeline if any step fails.")

    args = parser.parse_args()

    try:
        if args.step == "all":
            for name, func in PIPELINE_STEPS.items():
                logging.info(f"Running step: {name}")
                try:
                    func()
                    logging.info(f"Step {name} completed successfully.")
                except Exception as e:
                    logging.error(f"Error in step {name}: {e}")
                    if args.stop_on_fail:
                        raise e
        elif args.step == "eval":
            for name, func in EVAL_STEPS.items():
                logging.info(f"Running evaluation step: {name}")
                func()
        elif args.step in PIPELINE_STEPS:
            PIPELINE_STEPS[args.step]()
        elif args.step in EVAL_STEPS:
            EVAL_STEPS[args.step]()
        else:
            parser.print_help()
    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        if args.stop_on_fail:
            sys.exit(1)

if __name__ == "__main__":
    main()


