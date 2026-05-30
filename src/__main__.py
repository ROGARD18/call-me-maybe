import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.models import FunctionsClass
from src.vocab_dict import make_vocab_dict


def generate_with_constraints(
    prompt_text: str, 
    functions_class: FunctionsClass, 
    llm: Small_LLM_Model, 
    vocab: Dict[str, int], 
    inverse_vocab: Dict[int, str]
) -> Dict[str, Any]:
    
    # 1. Schéma propre en texte
    schema_lines = []
    for fn_name, fn_def in functions_class.definitions.items():
        params = fn_def.get("parameters", {})
        param_strs = [f'"{k}"' for k in params.keys()]
        reqs = ", ".join(param_strs) if param_strs else "no parameters"
        schema_lines.append(f"- {fn_name}: requires {reqs}")
    defs_str = "\n".join(schema_lines)

    prefix_str = f'{{\n  "prompt": "{prompt_text}",\n  "name": "'
    context = (
        "Extract the required function parameters as a JSON object.\n"
        f"Available functions:\n{defs_str}\n\n"
        f"Request: {prompt_text}\n"
        f"Output:\n{prefix_str}"
    )
    
    generated_tokens = llm.encode(context).tolist()[0]
    
    state = "NAME"
    selected_function_name = ""
    current_text = prefix_str
    params_text = ""
    
    print(prefix_str, end="", flush=True)

    for _ in range(300): 
        # TRANSITION
        if state == "TRANSITION":
            transition_str = '",\n  "parameters": '
            generated_tokens.extend(llm.encode(transition_str).tolist()[0])
            print(transition_str, end="", flush=True)
            current_text += transition_str
            state = "PARAMS"
            params_text = "" # On réinitialise pour capturer UNIQUEMENT le JSON des paramètres
            continue
            
        logits = llm.get_logits_from_input_ids(generated_tokens)
        logits_np = np.array(logits, dtype=np.float32)
        is_allowed = np.zeros(len(logits_np), dtype=bool)
        
        # CONTRAINTES
        if state == "NAME":
            name_start = current_text.rfind('"name": "') + 9
            current_name = current_text[name_start:]
            
            if current_name in functions_class.list:
                for tid, tstr in inverse_vocab.items():
                    if '"' in tstr:
                        is_allowed[tid] = True
                        
            for tid, tstr in inverse_vocab.items():
                clean = tstr.replace("Ġ", "").replace("Ċ", "").replace("ĉ", "")
                if not clean:
                    continue
                potential = current_name + clean
                if any(fn.startswith(potential) for fn in functions_class.list):
                    is_allowed[tid] = True
                    
        elif state == "PARAMS":
            # Analyse de ce qui a été généré DANS les paramètres
            b_count = 0
            i_str = False
            esc = False
            h_opened = False
            found_keys = set()
            curr_str = ""

            for i, char in enumerate(params_text):
                if esc:
                    esc = False
                    continue
                if char == '\\':
                    esc = True
                    continue
                if char == '"':
                    i_str = not i_str
                    if not i_str:
                        # Détection des clés (suivies de ':')
                        j = i + 1
                        while j < len(params_text) and params_text[j] in ' \n\t\r':
                            j += 1
                        if j < len(params_text) and params_text[j] == ':':
                            found_keys.add(curr_str)
                    curr_str = ""
                    continue

                if i_str:
                    curr_str += char
                else:
                    if char == '{':
                        h_opened = True
                        b_count += 1
                    elif char == '}':
                        b_count -= 1

            expected_keys = functions_class.definitions.get(selected_function_name, {}).get("parameters", {}).keys()
            all_keys_found = (len(found_keys) >= len(expected_keys))

            if not h_opened:
                for tid, tstr in inverse_vocab.items():
                    clean = tstr.replace("Ġ", "").replace("Ċ", "").replace("ĉ", "")
                    if "{" in clean and "[" not in clean:
                        is_allowed[tid] = True
            elif i_str:
                is_allowed.fill(True)
            else:
                # LA CORRECTION EST ICI : on autorise TOUS les caractères valides du JSON y compris les sauts de ligne \n \t \r
                valid_chars = set(' {}[],.:"-_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\n\t\r\\/')
                for tid, tstr in inverse_vocab.items():
                    clean = tstr.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
                    if all(c in valid_chars for c in clean):
                        # Blocage des virgules si on a déjà toutes les clés (force la fermeture)
                        if all_keys_found and ',' in clean:
                            continue
                        is_allowed[tid] = True

        if not np.any(is_allowed):
            is_allowed.fill(True)
            
        # Triage du Token (Greedy Decoding)
        masked_logits = np.where(is_allowed, logits_np, -1e10)
        next_token = int(np.argmax(masked_logits))
        
        generated_tokens.append(next_token)
        token_str = inverse_vocab.get(next_token, "")
        clean_new_token = token_str.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
        
        print(clean_new_token, end="", flush=True)
        
        current_text += clean_new_token
        if state == "PARAMS":
            params_text += clean_new_token

        # VERIFICATION DES ÉTATS ET ARRÊT
        if state == "NAME":
            name_start = current_text.rfind('"name": "') + 9
            current_name = current_text[name_start:]
            if current_name.endswith('"'):
                func_name = current_name[:-1]
                if func_name in functions_class.list:
                    selected_function_name = func_name
                    state = "TRANSITION"
                    
        elif state == "PARAMS":
            # On recompte rapidement pour voir si le token qu'on vient d'ajouter a fermé l'accolade
            final_b = 0
            final_str = False
            final_esc = False
            final_open = False
            for char in params_text:
                if final_esc:
                    final_esc = False
                    continue
                if char == '\\':
                    final_esc = True
                    continue
                if char == '"':
                    final_str = not final_str
                    continue
                if not final_str:
                    if char == '{':
                        final_open = True
                        final_b += 1
                    elif char == '}':
                        final_b -= 1

            # CONDITIONS D'ARRÊT PARFAITES
            if final_open and final_b <= 0 and not final_str:
                print("\n", flush=True)
                break
                
    # L'extraction ne peut plus échouer, on parse directement params_text
    try:
        params_obj = json.loads(params_text)
    except Exception:
        params_obj = {}

    return {
        "prompt": prompt_text,
        "name": selected_function_name,
        "parameters": params_obj
    }


def load_json_file(file_path: str) -> Any:
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained Decoding Function Caller")
    parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str, default="data/output/function_calling_results.json")
    args = parser.parse_args()

    f_data = load_json_file(args.functions_definition)
    names = [f.get("name") for f in f_data]
    defs = {f["name"]: f for f in f_data if isinstance(f, dict) and "name" in f}
    functions_class = FunctionsClass(list=names, definitions=defs)

    input_data = load_json_file(args.input)

    llm = Small_LLM_Model()
    vocab = make_vocab_dict(llm.get_path_to_vocabulary_json())
    inverse_vocab = {int(v): k for k, v in vocab.items()}

    results = []
    for i, item in enumerate(input_data):
        prompt_text = item["prompt"]
        print(f"\n--- Processing prompt {i + 1}/{len(input_data)} ---")
        
        try:
            result_json = generate_with_constraints(
                prompt_text, functions_class, llm, vocab, inverse_vocab
            )
            results.append(result_json)
        except Exception as e:
            print(f"\n[!] Error processing prompt {i}: {e}", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print("\n--- TERMINE ! ---")


if __name__ == "__main__":
    main()