"""GSM8K via genlm-eval. The paper's prompt pair (App. J.4) is ``few_shot_cot`` (system prompt plus
Q/A chain-of-thought examples) / ``precision_cot`` (step-by-step instruction); ``few_shot`` /
``direct`` are genlm-eval's plain formats, used by earlier runs and kept for reference."""

from genlm.eval.domains.gsm8k.gsm8k import (
    GSM8KDataset,
    GSM8KEvaluator,
    chain_of_thought_prompt_formatter,
    default_prompt_formatter,
    direct_answer_prompt_formatter,
    few_shot_prompt_formatter,
)

_FEW_SHOT_COT_EXAMPLES = """\
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
"""

_FEW_SHOT_COT_EXAMPLES_2 = """\
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
"""

_PRECISION_COT_INSTRUCTION = (
    "You are a precise mathematical reasoner. Your goal is correctness, not speed.\n\n"
    "For each problem, follow these steps:\n"
    "1. Extract every numerical fact stated in the problem — nothing more, nothing less.\n"
    "2. State exactly what quantity is being asked for.\n"
    "3. Determine the sequence of operations needed, giving a reason for each.\n"
    "4. Carry out each arithmetic step explicitly.\n"
    "5. State your answer as \"The answer is X.\""
)

_ZEROSHOT_COT_INSTRUCTION = (
    "Answer the following math problem. "
    "Before computing anything, identify: (1) what quantity is being asked for, "
    "(2) what information is given, and (3) how they relate. "
    "Then work through the solution step by step. "
    "At the end, verify your answer makes sense given the problem."
)


def _chat_or_plain(tokenizer, system, user, plain, use_chat_format):
    if use_chat_format:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True,
        )
    return tokenizer.encode(plain)


def _few_shot_cot(examples):
    def fmt(tokenizer, instance, use_chat_format=False):
        return _chat_or_plain(
            tokenizer,
            "You are a helpful assistant that solves grade school math word problems. Show your reasoning step by step, then state the final answer.",
            examples + f"Q: {instance.question}\nA:", examples + f"Q: {instance.question}\nA:", use_chat_format,
        )
    return fmt


def zeroshot_cot_prompt_formatter(tokenizer, instance, use_chat_format=False):
    return _chat_or_plain(tokenizer, _ZEROSHOT_COT_INSTRUCTION, f"Problem: {instance.question}\n\nSolution:",
                          f"{_ZEROSHOT_COT_INSTRUCTION}\n\nProblem: {instance.question}\n\nSolution:", use_chat_format)


def precision_cot_prompt_formatter(tokenizer, instance, use_chat_format=False):
    return _chat_or_plain(tokenizer, _PRECISION_COT_INSTRUCTION, f"Problem: {instance.question}",
                          f"{_PRECISION_COT_INSTRUCTION}\n\nProblem: {instance.question}", use_chat_format)


FORMATTERS = {
    "default": default_prompt_formatter,
    "direct": direct_answer_prompt_formatter,
    "cot": chain_of_thought_prompt_formatter,
    "few_shot": few_shot_prompt_formatter,
    "few_shot_cot": _few_shot_cot(_FEW_SHOT_COT_EXAMPLES),
    "few_shot_cot_2": _few_shot_cot(_FEW_SHOT_COT_EXAMPLES_2),
    "zeroshot_cot": zeroshot_cot_prompt_formatter,
    "precision_cot": precision_cot_prompt_formatter,
}


class Task:
    name = "gsm8k"
    templates = ("few_shot_cot", "precision_cot")
    max_tokens = {"token": 1024, "byte": 1024}
    max_model_len = 4096
    temperature = 0.7  # the paper's GSM8K setting (App. G)

    def __init__(self, n_query=100, base_seed=42, split="test"):
        # Same as GSM8KDataset.from_hf(split, max_instances=n_query, shuffle=False), except that
        # genlm-eval asks the hub for the bare id "gsm8k", which current huggingface_hub rejects
        # ("Repository id must be 'namespace/name'"); the canonical id is openai/gsm8k.
        from datasets import load_dataset

        rows = list(load_dataset("openai/gsm8k", "main", split=split))
        if n_query is not None:
            rows = rows[:n_query]
        self.items = list(GSM8KDataset(rows))
        self.evaluator = GSM8KEvaluator()

    def prompt(self, item, template, tokenizer, slot=1):
        name = (getattr(tokenizer, "name_or_path", "") or "").lower()
        use_chat = "instruct" in name or "chat" in name
        return tokenizer.decode(FORMATTERS[template](tokenizer, item, use_chat_format=use_chat))

    def score(self, item, text):
        r = self.evaluator.evaluate_sample(item, text)
        return float(r.score) >= 1.0, r.desc

    def record(self, item):
        return {"question": item.question, "gold": item.answer}
