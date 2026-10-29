if __package__:
    from .config import parse_builder_args
else:
    from config import parse_builder_args

ACTION_TOKENS = [
    "FORWARD_WALK", "DASH", "BACK_STEP", "CROUCH", "JUMP", "FOR_JUMP",
    "BACK_JUMP", "STAND_GUARD", "CROUCH_GUARD", "AIR_GUARD", "STAND_A", "STAND_B",
    "THROW_A", "THROW_B", "CROUCH_A", "CROUCH_B", "STAND_FA", "STAND_FB", "CROUCH_FA",
    "CROUCH_FB", "STAND_F_D_DFA", "STAND_F_D_DFB", "STAND_D_DB_BA", "STAND_D_DB_BB",
    "STAND_D_DF_FA", "STAND_D_DF_FB", "STAND_D_DF_FC", "AIR_A", "AIR_B", "AIR_DA",
    "AIR_DB", "AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "AIR_F_D_DFA", "AIR_F_D_DFB",
    "AIR_D_DB_BA", "AIR_D_DB_BB", "AIR_D_DF_FA", "AIR_D_DF_FB",
]

def main(argv=None):
    args = parse_builder_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    existing_vocab = tokenizer.get_vocab()
    tokens_to_add = [tok for tok in ACTION_TOKENS if tok not in existing_vocab]

    print(f"New tokens to add: {len(tokens_to_add)}")
    if tokens_to_add:
        added = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        print(f"Tokenizer added {added} tokens.")
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    print("\nTokenization check:")
    bad = []
    for tok in ACTION_TOKENS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        print(f"{tok:20s} -> {ids} len={len(ids)}")
        if len(ids) != 1:
            bad.append((tok, ids))

    print(f"\nSaved to: {args.output_dir.resolve()}")
    if bad:
        print("\nWARNING: Some actions are still multi-token:")
        for tok, ids in bad:
            print(tok, ids)
    else:
        print("\nAll legal actions are single-token now.")

if __name__ == "__main__":
    main()