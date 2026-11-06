"""Big-Bench Hard, word sorting. Prompts and scorer are the paper's, unchanged."""

import logging

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)


def _extract_words(word_data):
    """Return the words as a comma-separated string (the dataset wraps them in 'List: ...')."""
    if isinstance(word_data, list):
        if len(word_data) == 1 and isinstance(word_data[0], str) and "List:" in word_data[0]:
            words_str = word_data[0].split("List:")[-1].strip()
            return ", ".join(w.strip() for w in words_str.split() if w.strip())
        return ", ".join(w.strip() for w in word_data[0].split() if w.strip())
    if isinstance(word_data, str):
        if "List:" in word_data:
            word_data = word_data.split("List:")[-1].strip()
        return ", ".join(w.strip() for w in word_data.split() if w.strip())
    return str(word_data)


# the paper's pair is instructional / example_based; the other four are the K-expert ablation's
SYSTEM_PROMPTS = {
    "instructional": (
        "You are a helpful assistant that sorts words in alphabetical order. "
        "Your task is to take a list of words and return them sorted alphabetically. "
        "Alphabetical order means sorting words from A to Z, comparing character by character from left to right. "
        "For example, 'apple' comes before 'banana', and 'cat' comes before 'dog'. "
        "When words start with the same letter, compare the next letters. "
        "Output ONLY the sorted words separated by commas, with no additional text, explanation, or commentary. "
        "Do not include any prefixes, suffixes, or extra formatting - just the comma-separated sorted words."
    ),
    "example_based": (
        "You are a helpful assistant that sorts words in alphabetical order. "
        "Your task is to take a list of words and return them sorted alphabetically. "
        "Output ONLY the sorted words separated by commas, with no additional text, explanation, or commentary."
    ),
    "minimal": "Sort the words alphabetically. Output only the comma-separated sorted words.",
    "detailed": (
        "You are a meticulous assistant specialized in alphabetical sorting. Given a list of words, "
        "return them ordered from A to Z. Compare words character by character from left to right; when a "
        "prefix is shared, the shorter word or the one whose next character is earlier in the alphabet comes "
        "first. Treat the comparison as case-insensitive. Produce every input word exactly once, with none "
        "added or dropped. Respond with ONLY the sorted words joined by commas - no numbering, quotes, "
        "explanation, or surrounding text."
    ),
    "technical": (
        "Task: alphabetical sort of a word list.\n"
        "Input: comma-separated words.\n"
        "Ordering: lexicographic, A-Z, left-to-right character comparison, case-insensitive.\n"
        "Output: the input words reordered, comma-separated, single line, no extra text."
    ),
    "persona": (
        "You are SORT(), a deterministic alphabetical-ordering function. You take a list of words and "
        "return exactly the same words rearranged into A-to-Z order. You emit only the comma-separated "
        "result, never commentary."
    ),
}


def format_word_sorting_prompt(words, tokenizer, prompt_template="instructional", few_shot_examples=None):
    if prompt_template not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt template: {prompt_template}. Must be one of {sorted(SYSTEM_PROMPTS)}")
    system_prompt = SYSTEM_PROMPTS[prompt_template]

    is_gemma = "gemma" in (getattr(tokenizer, "name_or_path", "") or "").lower()
    messages = [] if is_gemma else [{"role": "system", "content": system_prompt}]

    if prompt_template == "example_based":
        for example in few_shot_examples or []:
            messages.append({"role": "user", "content": f"{_extract_words(example['input'])} ->"})
            messages.append({"role": "assistant", "content": _extract_words(example["target"])})
        messages.append({"role": "user", "content": f"{_extract_words(words)} ->"})
    else:  # instruction-style Input/Output turn for every other template
        messages.append({"role": "user", "content": f"Sort the following words in alphabetical order.\n\nInput: {_extract_words(words)}\nOutput:"})

    if is_gemma:  # no system role: fold everything into one user turn
        user_content = system_prompt + "\n\n" + "".join(m["content"] + "\n" for m in messages)
        messages = [{"role": "user", "content": user_content}]

    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        logger.warning(f"Failed to apply chat template: {e}, using plain format")
        return "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in messages)


def evaluate_word_sorting(pred: str, gold: str):
    def normalize_words(s):
        words = s.split(",") if "," in s else s.split()
        return [w.strip().lower() for w in words if w.strip()]

    pred_words, gold_words = normalize_words(pred), normalize_words(gold)
    if len(pred_words) != len(gold_words):
        return False, "length_mismatch"
    return (True, "correct") if pred_words == gold_words else (False, "mismatch")


class Task:
    name = "bbh"
    templates = ("instructional", "example_based")
    max_tokens = {"token": 256, "byte": 256}
    max_model_len = 2048
    temperature = 1.0

    def __init__(self, n_query=100, base_seed=42, few_shot=0):
        dataset = list(load_dataset("maveriq/bigbenchhard", "word_sorting", split="train", trust_remote_code=True))
        if n_query is None or n_query >= len(dataset):
            eval_data = dataset
        else:
            # the paper's runner subsampled with a fixed RandomState(42), independent of base_seed
            indices = sorted(np.random.RandomState(42).choice(len(dataset), size=n_query, replace=False))
            eval_data = [dataset[i] for i in indices]

        self.items = []
        for ex in eval_data:
            inp = ex.get("inputs", ex.get("input"))
            tgt = ex.get("targets", ex.get("target"))
            if isinstance(inp, str):
                inp = [w.strip() for w in inp.split(",")]
            if isinstance(tgt, str):
                tgt = [w.strip() for w in tgt.split(",")]
            self.items.append({"input_words": inp, "target_words": tgt, "gold": ", ".join(tgt)})

        self.few_shot_examples = None
        if few_shot > 0:
            idx = np.random.RandomState(42).choice(len(dataset), size=min(few_shot, len(dataset)), replace=False)
            self.few_shot_examples = [
                {"input": [w.strip() for w in dataset[i].get("inputs", dataset[i].get("input")).split(",")],
                 "target": [w.strip() for w in dataset[i].get("targets", dataset[i].get("target")).split(",")]}
                for i in idx
            ]

    def prompt(self, item, template, tokenizer, slot=1):
        return format_word_sorting_prompt(item["input_words"], tokenizer, template, self.few_shot_examples)

    def score(self, item, text):
        return evaluate_word_sorting(text, item["gold"])

    def record(self, item):
        return {"input": ", ".join(item["input_words"]), "gold": item["gold"]}
