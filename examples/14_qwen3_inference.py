"""Qwen3-0.6B generation with all Linear layers mapped to arrays."""
import torch

from _common import engine, parser, report
from memintelli import convert_model


def main():
    p = parser(__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="Explain a memristor in one sentence.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    simulator = engine(args) if not args.digital else None
    if simulator is not None:
        model = convert_model(model, simulator)
    text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(args.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    report(args, {"model": args.model, "generated_text": answer}, simulator)


if __name__ == "__main__":
    main()
