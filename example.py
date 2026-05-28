import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0125-Instruct/snapshots/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=2)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "where is india located?"
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
