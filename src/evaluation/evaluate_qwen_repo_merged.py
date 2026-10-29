import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ACTIONS = [
    "FORWARD_WALK", "DASH", "BACK_STEP", "CROUCH", "JUMP", "FOR_JUMP",
    "BACK_JUMP", "STAND_GUARD", "CROUCH_GUARD", "AIR_GUARD", "STAND_A", "STAND_B",
    "THROW_A", "THROW_B", "CROUCH_A", "CROUCH_B", "STAND_FA", "STAND_FB",
    "CROUCH_FA", "CROUCH_FB", "STAND_F_D_DFA", "STAND_F_D_DFB",
    "STAND_D_DB_BA", "STAND_D_DB_BB", "STAND_D_DF_FA", "STAND_D_DF_FB",
    "STAND_D_DF_FC", "AIR_A", "AIR_B", "AIR_DA", "AIR_DB", "AIR_FA",
    "AIR_FB", "AIR_UA", "AIR_UB", "AIR_F_D_DFA", "AIR_F_D_DFB",
    "AIR_D_DB_BA", "AIR_D_DB_BB", "AIR_D_DF_FA", "AIR_D_DF_FB",
]

NON_ATTACK_ACTIONS = {
    "FORWARD_WALK", "DASH", "BACK_STEP", "CROUCH", "JUMP", "FOR_JUMP",
    "BACK_JUMP", "STAND_GUARD", "CROUCH_GUARD", "AIR_GUARD",
}

ATTACK_ACTIONS = set(ACTIONS) - NON_ATTACK_ACTIONS

DEFAULT_INSTRUCTION = "Predict the next FightingICE action."


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def row_to_fields(row: dict) -> tuple[str, str, str]:
    if "messages" in row:
        messages = row["messages"]
        instruction = messages[0]["content"].strip() if messages else DEFAULT_INSTRUCTION
        input_text = messages[1]["content"].strip()
        output = messages[2]["content"].strip().upper()
        return instruction, input_text, output

    instruction = row.get("instruction", DEFAULT_INSTRUCTION).strip()
    input_value = row["input"]
    input_text = (
        input_value
        if isinstance(input_value, str)
        else json.dumps(input_value, ensure_ascii=False, separators=(",", ":"))
    )
    output = row["output"].strip().upper()
    return instruction, input_text, output


def build_prompt(instruction: str, input_text: str) -> str:
    return f"Instruction:{instruction}\nInput:{input_text}\nOutput:\n"


def verify_action_tokens(tokenizer) -> list[int]:
    token_ids = []
    bad = []
    for action in ACTIONS:
        ids = tokenizer.encode(action, add_special_tokens=False)
        if len(ids) != 1:
            bad.append((action, ids))
        else:
            token_ids.append(ids[0])
    if bad:
        raise RuntimeError(f"Some actions are not single-token: {bad}")
    return token_ids


def decode_one(tokenizer, token_ids) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False).strip()


@torch.no_grad()
def evaluate(model, tokenizer, rows: list[dict], batch_size: int, device: torch.device) -> dict:
    model.eval()
    action_token_ids = torch.tensor(verify_action_tokens(tokenizer), dtype=torch.long, device=device)

    exact = 0
    valid = 0
    top_exact = 0
    pred_counts = Counter()
    top_counts = Counter()
    gold_counts = Counter()
    confusion = Counter()
    examples = []

    prompts = []
    golds = []
    for row in rows:
        instruction, input_text, gold = row_to_fields(row)
        prompts.append(build_prompt(instruction, input_text))
        golds.append(gold)

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_golds = golds[start:start + batch_size]

        enc = tokenizer(
            batch_prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        raw = model(**enc)
        last_indices = enc["attention_mask"].sum(dim=1) - 1
        next_logits = raw.logits[torch.arange(raw.logits.shape[0], device=device), last_indices, :]
        action_logits = next_logits.index_select(dim=1, index=action_token_ids)
        top_indices = action_logits.argmax(dim=1).detach().cpu().tolist()
        top_actions = [ACTIONS[i] for i in top_indices]

        generated = model.generate(
            **enc,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = enc["input_ids"].shape[1]
        for i, gold in enumerate(batch_golds):
            pred = decode_one(tokenizer, generated[i][prompt_len:])
            top_action = top_actions[i]

            gold_counts[gold] += 1
            pred_counts[pred] += 1
            top_counts[top_action] += 1
            confusion[(gold, pred)] += 1

            if pred == gold:
                exact += 1
            if pred in ACTIONS:
                valid += 1
            if top_action == gold:
                top_exact += 1

            if len(examples) < 20 and pred != gold:
                examples.append({
                    "idx": start + i + 1,
                    "gold": gold,
                    "pred": pred,
                    "top_action": top_action,
                    "prompt": batch_prompts[i][:240],
                })

    total = len(rows)
    attack_preds = sum(count for action, count in pred_counts.items() if action in ATTACK_ACTIONS)
    non_attack_preds = sum(count for action, count in pred_counts.items() if action in NON_ATTACK_ACTIONS)
    invalid_preds = total - valid

    return {
        "total": total,
        "exact": exact,
        "valid": valid,
        "top_exact": top_exact,
        "attack_preds": attack_preds,
        "non_attack_preds": non_attack_preds,
        "invalid_preds": invalid_preds,
        "pred_counts": pred_counts,
        "top_counts": top_counts,
        "gold_counts": gold_counts,
        "confusion": confusion,
        "examples": examples,
    }


def print_counter(title: str, counts: Counter, total: int, limit: int = 40) -> None:
    print(f"\n=== {title} ===")
    for i, (item, count) in enumerate(counts.most_common(limit), 1):
        pct = count / total * 100 if total else 0
        print(f"{i:2d}) {item:20s} {count:6d} {pct:7.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen_ftg_repo_merged")
    parser.add_argument("--data", default="ftg_messages_plain/valid.jsonl")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--load_4bit", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        raise FileNotFoundError(f"Model folder not found: {model_path.resolve()}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {model_path}")
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    if args.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    rows = load_jsonl(data_path, args.limit)
    print(f"Loaded rows: {len(rows)}")

    stats = evaluate(model, tokenizer, rows, args.batch_size, device)
    total = stats["total"]

    print("\n=== SUMMARY ===")
    print(f"Total:                  {total}")
    print(f"Exact match:            {stats['exact']}/{total} = {stats['exact'] / total:.4f}")
    print(f"Valid generated action: {stats['valid']}/{total} = {stats['valid'] / total:.4f}")
    print(f"Top legal exact match:  {stats['top_exact']}/{total} = {stats['top_exact'] / total:.4f}")
    print(f"Attack predictions:     {stats['attack_preds']}/{total} = {stats['attack_preds'] / total:.4f}")
    print(f"Non-attack predictions: {stats['non_attack_preds']}/{total} = {stats['non_attack_preds'] / total:.4f}")
    print(f"Invalid predictions:    {stats['invalid_preds']}/{total} = {stats['invalid_preds'] / total:.4f}")

    print_counter("GOLD DISTRIBUTION", stats["gold_counts"], total)
    print_counter("GENERATED PRED DISTRIBUTION", stats["pred_counts"], total)
    print_counter("TOP LEGAL ACTION DISTRIBUTION", stats["top_counts"], total)

    print("\n=== COMMON CONFUSIONS ===")
    for i, ((gold, pred), count) in enumerate(stats["confusion"].most_common(30), 1):
        if gold == pred:
            continue
        print(f"{i:2d}) gold={gold:20s} pred={pred:20s} count={count}")

    if stats["examples"]:
        print("\n=== SAMPLE MISMATCHES ===")
        for ex in stats["examples"]:
            print(
                f"idx={ex['idx']} gold={ex['gold']} pred={ex['pred']} "
                f"top_action={ex['top_action']}"
            )


if __name__ == "__main__":
    main()
