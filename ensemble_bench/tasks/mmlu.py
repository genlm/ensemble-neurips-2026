"""MMLU. The paper's prompts (App. J.5) are ``template1_text_with_answers`` / ``template2_text_with_answers``:
the model answers with the *text* of the correct option, scored by normalised match against the
options. ``template1`` / ``template2`` are the letter-answer variants (A-D), kept for reference.

Few-shot examples (5, subject-matched, from the dev split) go into every template except the
letter-answer ``template1`` -- the paper's runner tested the template name for equality with
"template1", so both text-with-answers prompts are 5-shot, with different draws per slot.
"""

import logging
import re
from typing import Optional

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)

NUM_FEWSHOT = 5
SYSTEM_PROMPTS = {
    "letter": (
        "You are a helpful assistant that answers multiple choice questions. "
        "Respond with ONLY the letter of your answer (A, B, C, or D). "
        "Do not include any explanation or additional text."
    ),
    "text": (
        "You are a helpful assistant that answers multiple choice questions. "
        "Respond with ONLY the exact text of the correct answer choice. "
        "Do not include any explanation or additional text."
    ),
}


def output_mode(template):
    return "text" if "_text_" in template else "letter"


def format_mmlu_prompt(question, choices, tokenizer, subject=None, few_shot_examples=None, template="template1"):
    mode = output_mode(template)
    system_prompt = SYSTEM_PROMPTS[mode]
    header = "answers" if mode == "text" else "answer letters"
    is_gemma = "gemma" in (getattr(tokenizer, "name_or_path", "") or "").lower()
    subject_name = subject.replace("_", " ").lower() if subject and subject != "all" else None

    def shown_answer(example):
        return example["choices"][example["answer"]] if mode == "text" else chr(65 + example["answer"])

    messages = [] if is_gemma else [{"role": "system", "content": system_prompt}]
    parts = [system_prompt, ""] if is_gemma else []

    if template.startswith("template1"):
        parts.append(
            f"The following are multiple choice questions (WITH {header}) about {subject_name}'"
            if subject_name else f"The following are multiple choice questions (WITH {header})'"
        )
        for example in few_shot_examples or []:
            text = f"\n{example['question']}\nAnswers'\n"
            text += "".join(f"{chr(65 + i)}. {c}\n" for i, c in enumerate(example["choices"]))
            text += f"Answer'\n{shown_answer(example)}"
            parts.append(text)
        text = f"\n{question}\nAnswers'\n" + "".join(f"{chr(65 + i)}. {c}\n" for i, c in enumerate(choices)) + "Answer'\n"
        parts.append(text)
    elif template.startswith("template2"):
        parts.append(
            f"The following are multiple choice questions (With {header}) about {subject_name}<sep>"
            if subject_name else f"The following are multiple choice questions (With {header})<sep>"
        )
        for example in few_shot_examples or []:
            text = f"{example['question']}<sep>\nAnswers:\n"
            text += "".join(f"{chr(65 + i)}. {c}<sep>\n" for i, c in enumerate(example["choices"]))
            text += f"Answer:\n{shown_answer(example)}"
            parts.append(text)
        text = f"{question}<sep>\nAnswers:\n" + "".join(f"{chr(65 + i)}. {c}<sep>\n" for i, c in enumerate(choices)) + "Answer:\n"
        parts.append(text)
    else:
        raise ValueError(f"Unknown template: {template}")

    messages.append({"role": "user", "content": "\n\n".join(parts)})
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception as e:
        logger.warning(f"Failed to apply chat template: {e}, using plain format")
        return messages[-1]["content"]


def _normalize(s: str) -> str:
    """Option text normalisation of the paper's scorer: <sep>, quotes, whitespace, case, trailing punctuation."""
    s = (s or "").strip().replace("<sep>", " ")
    s = re.sub(r"""^['"]|['"]$""", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip().lower()
    return re.sub(r"[.!?]+$", "", s).strip()


def extract_answer_text_choice(predicted: str, choices) -> Optional[str]:
    """The option whose text the model produced: the last ``Answer:`` field or the first line, matched
    exactly or by containment (longer option preferred)."""
    text = (predicted or "").strip()
    match = re.findall(r"(?:Answer|answer)\s*[:'\"]\s*(.+)", text)
    if match:
        text = match[-1]
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            text = lines[0]
    pred = _normalize(text)
    if not pred:
        return None
    best, best_len = None, -1
    for choice in choices:
        norm = _normalize(choice)
        if not norm:
            continue
        if pred == norm:
            return choice
        if pred in norm or norm in pred:
            if len(norm) > best_len:
                best, best_len = choice, len(norm)
    return best


def extract_answer_choice(text: str) -> Optional[str]:
    text = text.strip()
    m = re.match(r"^\s*([A-D])\.?\s*", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:Answer|answer|The answer is|The correct answer is)[:\s]+([A-D])\.?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


class Task:
    name = "mmlu"
    templates = ("template1_text_with_answers", "template2_text_with_answers")
    max_tokens = {"token": 1024, "byte": 1024}   # the paper's budget; answers are a few tokens
    max_model_len = 4096
    temperature = 1.0

    def __init__(self, n_query=100, base_seed=42):
        self.base_seed = base_seed
        data = list(load_dataset("cais/mmlu", "all", split="test"))
        self.dev = list(load_dataset("cais/mmlu", "all", split="dev"))
        if n_query is not None and n_query < len(data):
            np.random.seed(base_seed)
            data = [data[i] for i in np.random.choice(len(data), size=n_query, replace=False)]
        self.items = [dict(it, correct_choice=chr(65 + it["answer"])) for it in data]

    def _few_shot(self, item, seed):
        subj = item["subject"].lower().replace(" ", "_")
        pool = [ex for ex in self.dev if ex["subject"].lower().replace(" ", "_") == subj]
        if len(pool) < NUM_FEWSHOT:
            raise ValueError(f"Not enough few-shot examples for subject {item['subject']!r}: {len(pool)}")
        np.random.seed(seed)  # the paper's runner reseeded per example; kept for identical draws
        idx = np.random.choice(len(pool), size=NUM_FEWSHOT, replace=False)
        return [{"question": pool[j]["question"], "choices": pool[j]["choices"], "answer": pool[j]["answer"]} for j in idx]

    def prompt(self, item, template, tokenizer, slot=1):
        few_shot = None
        if template != "template1":  # exact match, as in the paper's runner
            # slot 2 draws with base_seed, slot 1 with base_seed + 1000
            few_shot = self._few_shot(item, self.base_seed + (1000 if slot == 1 else 0))
        return format_mmlu_prompt(item["question"], item["choices"], tokenizer, subject=item["subject"],
                                  few_shot_examples=few_shot, template=template)

    def score(self, item, text, template=None):
        """Letter mode: the A-D letter; text mode (the paper's): normalised match of the option text."""
        if output_mode(template or getattr(self, "score_template", None) or self.templates[0]) == "text":
            picked = extract_answer_text_choice(text, item["choices"])
            if picked is None:
                return False, None
            letter = chr(65 + item["choices"].index(picked))
            return _normalize(picked) == _normalize(item["choices"][item["answer"]]), letter
        choice = extract_answer_choice(text)
        return choice == item["correct_choice"], choice

    def record(self, item):
        return {"question": item["question"], "choices": item["choices"], "subject": item["subject"],
                "correct_choice": item["correct_choice"]}
