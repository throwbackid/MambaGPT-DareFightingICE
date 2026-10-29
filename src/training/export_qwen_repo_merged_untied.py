import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def copy_modules_to_save(model: PeftModel) -> None:
    copied = []

    for name, module in model.named_modules():
        if not hasattr(module, "modules_to_save"):
            continue
        if "default" not in module.modules_to_save:
            continue
        if not hasattr(module, "original_module"):
            print(f"WARNING: {name} has modules_to_save but no original_module; skipping")
            continue

        saved_module = module.modules_to_save["default"]
        original = module.original_module

        with torch.no_grad():
            if hasattr(saved_module, "weight") and hasattr(original, "weight"):
                if saved_module.weight.shape != original.weight.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {name}.weight: "
                        f"saved={tuple(saved_module.weight.shape)} "
                        f"original={tuple(original.weight.shape)}"
                    )
                original.weight.copy_(
                    saved_module.weight.to(
                        device=original.weight.device,
                        dtype=original.weight.dtype,
                    )
                )
                copied.append(f"{name}.weight")

            if (
                hasattr(saved_module, "bias")
                and saved_module.bias is not None
                and hasattr(original, "bias")
                and original.bias is not None
            ):
                if saved_module.bias.shape != original.bias.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {name}.bias: "
                        f"saved={tuple(saved_module.bias.shape)} "
                        f"original={tuple(original.bias.shape)}"
                    )
                original.bias.copy_(
                    saved_module.bias.to(
                        device=original.bias.device,
                        dtype=original.bias.dtype,
                    )
                )
                copied.append(f"{name}.bias")

    if not copied:
        raise RuntimeError("Did not find any modules_to_save weights to copy.")

    print("Copied modules_to_save into base modules:")
    for item in copied:
        print(f"  - {item}")


def print_weight_status(model, label: str) -> None:
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    same_storage = emb.data_ptr() == head.data_ptr()
    print(
        f"{label}: emb_shape={tuple(emb.shape)} lm_head_shape={tuple(head.shape)} "
        f"same_storage={same_storage} tie_word_embeddings={getattr(model.config, 'tie_word_embeddings', None)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="qwen_ftg_repo_action_base")
    parser.add_argument("--adapter", default="qwen_ftg_repo_lora")
    parser.add_argument("--output", default="qwen_ftg_repo_merged_untied")
    args = parser.parse_args()

    base_dir = Path(args.base)
    adapter_dir = Path(args.adapter)
    output_dir = Path(args.output)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base folder not found: {base_dir.resolve()}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter folder not found: {adapter_dir.resolve()}")

    print(f"Loading tokenizer from: {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from: {base_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    print_weight_status(base, "Base before adapter")

    print(f"Loading adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)

    copy_modules_to_save(model)

    print("Merging LoRA weights...")
    merged = model.merge_and_unload()

    # Critical: do not call tie_weights(). The adapter learned separate
    # embed_tokens/lm_head modules_to_save. If the config remains tied, reload can
    # re-tie lm_head to embeddings and destroy the learned action-token head.
    merged.config.tie_word_embeddings = False
    print_weight_status(merged, "Merged before save")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving untied merged model to: {output_dir}")
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Done. Evaluate with: python evaluate_qwen_repo_merged.py --model {output_dir}")


if __name__ == "__main__":
    main()
