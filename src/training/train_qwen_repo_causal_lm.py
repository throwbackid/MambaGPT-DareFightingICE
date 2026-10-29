# Parse CLI options before importing optional GPU/model packages so --help works anywhere.
if __package__:
    from .config import parse_training_args, validate_training_paths
else:
    from config import parse_training_args, validate_training_paths

_cli_args = None
if __name__ == "__main__":
    _cli_args = parse_training_args()
    try:
        validate_training_paths(_cli_args)
    except ValueError as error:
        raise SystemExit(str(error)) from error

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =========================
# Config
# =========================
MAX_LENGTH = 384
VALID_BATCH_SIZE = 1
LEARNING_RATE = 2e-5

LOGGING_STEPS = 100
EVAL_STEPS = 1000
SAVE_STEPS = 1000
VAL_LIMIT = 50
GRAD_CLIP = 1.0

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_INSTRUCTION = (
    "You are the best and most aggressive player in a 2D fighting video game. "
    "You will receive the game state and must choose the best action."
)

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
# =========================


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_to_fields(row: dict):
    if "messages" in row:
        msgs = row["messages"]
        instruction = msgs[0]["content"].strip() if len(msgs) > 0 else DEFAULT_INSTRUCTION
        input_text = msgs[1]["content"].strip()
        output_text = msgs[2]["content"].strip()
        return instruction, input_text, output_text

    instruction = row.get("instruction", DEFAULT_INSTRUCTION).strip()
    input_text = row["input"]
    if not isinstance(input_text, str):
        input_text = json.dumps(input_text, separators=(",", ":"), ensure_ascii=False)
    output_text = row["output"].strip()
    return instruction, input_text, output_text


class FTGDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 384, limit: int | None = None):
        rows = load_jsonl(path)
        if limit is not None:
            rows = rows[:limit]
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        instruction, input_text, action = row_to_fields(self.rows[idx])

        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        prompt = (
            f"Instruction:{instruction}\n"
            f"Input:{input_text}\n"
            f"Output:\n"
        )
        answer = action + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        answer_ids = self.tokenizer(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = ([-100] * len(prompt_ids)) + answer_ids

        if len(input_ids) > self.max_length:
            keep_answer_len = len(answer_ids)
            remain_for_prompt = max(0, self.max_length - keep_answer_len)
            prompt_ids = prompt_ids[-remain_for_prompt:]
            input_ids = prompt_ids + answer_ids
            labels = ([-100] * len(prompt_ids)) + answer_ids

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_text": prompt,
            "gold_action": action,
        }


@dataclass
class TrainCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_n)
            attention_mask.append(x["attention_mask"] + [0] * pad_n)
            labels.append(x["labels"] + [-100] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class EvalCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch):
        prompt_features = [
            self.tokenizer(
                x["prompt_text"],
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_LENGTH,
            )["input_ids"]
            for x in batch
        ]
        max_len = max(len(ids) for ids in prompt_features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask = [], []
        prompt_texts, gold_actions = [], []

        for x, prompt_ids in zip(batch, prompt_features):
            pad_n = max_len - len(prompt_ids)
            input_ids.append(prompt_ids + [pad_id] * pad_n)
            attention_mask.append([1] * len(prompt_ids) + [0] * pad_n)
            prompt_texts.append(x["prompt_text"])
            gold_actions.append(x["gold_action"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_texts": prompt_texts,
            "gold_actions": gold_actions,
        }


def verify_action_tokens(tokenizer):
    print("\nVerifying action token encoding:", flush=True)
    all_ok = True
    for action in ACTIONS:
        ids = tokenizer.encode(action, add_special_tokens=False)
        if len(ids) == 1:
            print(f"✓ {action} -> {ids[0]}", flush=True)
        else:
            print(f"✗ {action} -> {ids}", flush=True)
            all_ok = False
    if not all_ok:
        raise ValueError("Some actions are not single-token.")
    print("All action tokens are single-token.", flush=True)


@torch.no_grad()
def evaluate_generation(model, tokenizer, dataloader, device, max_examples=5):
    model.eval()

    action_token_ids = torch.tensor(
        [tokenizer.encode(action, add_special_tokens=False)[0] for action in ACTIONS],
        dtype=torch.long,
        device=device,
    )

    exact_match = 0
    valid_action = 0
    invalid = 0
    top_action_match = 0
    total = 0
    examples = []

    for batch in dataloader:
        model_inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        }

        raw_outputs = model(**model_inputs)
        last_indices = model_inputs["attention_mask"].sum(dim=1) - 1
        next_logits = raw_outputs.logits[
            torch.arange(raw_outputs.logits.shape[0], device=device),
            last_indices,
            :,
        ]
        action_logits = next_logits.index_select(dim=1, index=action_token_ids)
        top_action_indices = action_logits.argmax(dim=1).detach().cpu().tolist()
        top_actions = [ACTIONS[i] for i in top_action_indices]

        outputs = model.generate(
            **model_inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = model_inputs["input_ids"].shape[1]

        for i, gold in enumerate(batch["gold_actions"]):
            gen_ids = outputs[i][prompt_len:]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            top_action = top_actions[i]

            if pred == gold:
                exact_match += 1

            if pred in ACTIONS:
                valid_action += 1
            else:
                invalid += 1

            if top_action == gold:
                top_action_match += 1

            if len(examples) < max_examples:
                examples.append({
                    "idx": total + 1,
                    "gold": gold,
                    "pred": pred,
                    "top_action": top_action,
                    "valid": pred in ACTIONS,
                    "match": pred == gold,
                })

            total += 1

    model.train()

    return {
        "total": total,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / total if total else 0.0,
        "valid_action": valid_action,
        "valid_action_rate": valid_action / total if total else 0.0,
        "invalid": invalid,
        "invalid_rate": invalid / total if total else 0.0,
        "top_action_match": top_action_match,
        "top_action_match_rate": top_action_match / total if total else 0.0,
        "examples": examples,
    }


def save_adapter(model, tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def main(args=None):
    args = args or parse_training_args()
    validate_training_paths(args)
    print("SCRIPT START", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = "cuda"
    print(f"Using device: {device}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    verify_action_tokens(tokenizer)

    print("Preparing 4-bit quantization config...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading model in 4-bit...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        torch_dtype=torch.float16,
    )

    print("Preparing model for k-bit training...", flush=True)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    print("Applying LoRA...", flush=True)
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora)

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    print("Building datasets...", flush=True)
    train_dataset = FTGDataset(
        os.path.join(args.data_dir, "train.jsonl"),
        tokenizer,
        max_length=MAX_LENGTH,
    )
    valid_dataset = FTGDataset(
        os.path.join(args.data_dir, "valid.jsonl"),
        tokenizer,
        max_length=MAX_LENGTH,
        limit=VAL_LIMIT,
    )

    print(f"Train rows: {len(train_dataset)}", flush=True)
    print(f"Valid rows: {len(valid_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TrainCollator(tokenizer),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=EvalCollator(tokenizer),
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting repo-aligned causal LM training...", flush=True)
    model.train()

    step = 0
    running_loss = 0.0
    stop_training = False

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)

        for batch_index, batch in enumerate(train_loader, start=1):
            step += 1

            model_inputs = {
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
                "labels": batch["labels"].to(device, non_blocking=True),
            }

            outputs = model(**model_inputs)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

            if step % LOGGING_STEPS == 0:
                avg_loss = running_loss / LOGGING_STEPS
                print(f"Step {step}: Train loss {avg_loss:.4f}", flush=True)
                running_loss = 0.0

            if step % EVAL_STEPS == 0:
                stats = evaluate_generation(model, tokenizer, valid_loader, device)

                print(f"\nStep {step}: Validation summary", flush=True)
                print(
                    f"  Exact match: {stats['exact_match']}/{stats['total']} = {stats['exact_match_rate']:.4f}",
                    flush=True,
                )
                print(
                    f"  Valid action: {stats['valid_action']}/{stats['total']} = {stats['valid_action_rate']:.4f}",
                    flush=True,
                )
                print(
                    f"  Invalid:      {stats['invalid']}/{stats['total']} = {stats['invalid_rate']:.4f}",
                    flush=True,
                )
                print(
                    f"  Top legal action match: {stats['top_action_match']}/{stats['total']} = {stats['top_action_match_rate']:.4f}",
                    flush=True,
                )
                print("  Sample outputs:", flush=True)
                for ex in stats["examples"]:
                    print(
                        f"    idx={ex['idx']} gold={ex['gold']} pred={ex['pred']} "
                        f"top_action={ex['top_action']} valid={ex['valid']} match={ex['match']}",
                        flush=True,
                    )
                print("", flush=True)

            if step % SAVE_STEPS == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                print(f"Saving adapter to {ckpt_dir}", flush=True)
                save_adapter(model, tokenizer, ckpt_dir)

            if args.max_steps is not None and step >= args.max_steps:
                stop_training = True
                break

        # Keep completed epochs available for selection; a capped partial epoch
        # is saved only as the final adapter, not mislabeled as a full epoch.
        if not stop_training or batch_index == len(train_loader):
            epoch_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            print(f"Saving epoch adapter to {epoch_dir}", flush=True)
            save_adapter(model, tokenizer, epoch_dir)

        if stop_training:
            break

    print("Saving final adapter...", flush=True)
    save_adapter(model, tokenizer, args.output_dir)
    print(f"Done. Saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main(_cli_args)
