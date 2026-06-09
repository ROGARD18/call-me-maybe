import argparse
import json
import sys
from pathlib import Path

from src.constrained import generate_constrained_call
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.models import FunctionsClass
from src.vocab_dict import make_vocab_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input", type=str, default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/output/function_calling_results.json"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Lancer le mode interactif"
    )
    args = parser.parse_args()

    with open(args.functions_definition, "r", encoding="utf-8") as f:
        f_data = json.load(f)
    functions = FunctionsClass(
        list=[f["name"] for f in f_data], definitions={
            f["name"]: f for f in f_data}
    )

    llm = Small_LLM_Model()
    vocab = make_vocab_dict(llm.get_path_to_vocabulary_json())
    inverse_vocab = {int(v): k for k, v in vocab.items()}

    if args.interactive:
        while True:
            try:
                user_input = input("Prompt > ")
                if user_input.lower() in ("quit", "exit"):
                    break
                if not user_input.strip():
                    continue
                res = generate_constrained_call(
                    user_input, functions, llm, inverse_vocab
                )
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Erreur: {e}", file=sys.stderr)
        return

    with open(args.input, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    results = []
    for i, item in enumerate(prompts):
        print(f"\n--- Processing {i + 1}/{len(prompts)} ---")
        try:
            res = generate_constrained_call(
                item["prompt"], functions, llm, inverse_vocab
            )
            results.append(res)
        except Exception as e:
            print(f"Erreur: {e}", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
