import verifiers as vf
from datasets import load_dataset

def load_environment(
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    random_seed: int = 42,
) -> vf.Environment:
    """
    Defines and returns the PII Masking Environment.
    """

    # ===== System Prompt =====

    system_prompt = """Replace all personally identifiable information (PII) in the text with [PII] tags. 
PII includes: names, dates, phone numbers, SSNs, account numbers, addresses, email addresses, and any other identifying information.

Examples:
Input: Ticket Reservation for Florije: 'one ticket for Madame on October 8th, 1990'
Output: Ticket Reservation for [PII]: 'one ticket for [PII] on [PII]'

Input: User account recovery: "Hi Arljind Komla, your account recovery key is 426220045."
Output: User account recovery: "Hi [PII], your account recovery key is [PII]."

Return ONLY the masked text wrapped in masked_outputXML tags:
<masked_output>
[Your masked text here]
</masked_output>"""

    # ===== Dataset =====

    # Load Dataset
    ds_all = load_dataset("AdamLucek/open-pii-masking-en-us-30k")
    dataset = ds_all["train"]

    # Limit Training Examples if Specified
    if num_train_examples != -1:
        dataset = dataset.select(range(min(num_train_examples, len(dataset))))

    # Calculate eval size
    # Default to 20% of train dataset
    if num_eval_examples != -1:
        test_size = min(num_eval_examples, max(1, len(dataset) - 1))
    else:
        test_size = 0.2
    
    # Split Dataset
    split = dataset.train_test_split(test_size=test_size, seed=random_seed, shuffle=True)
    
    # Select Training and Evaluation Datasets from Split
    dataset = split["train"]
    eval_dataset = split["test"]

    # ===== Parser =====

    # Define Parser
    parser = vf.XMLParser(fields = ["masked_output"], answer_field = "masked_output")

    # ===== Reward Functions =====

    # Format Reward Function
    format_reward = parser.get_format_reward_func()

    # Exact Match Reward Function
    def exact_match_reward(parser, completion, answer) -> float:
        parsed_answer = parser.parse_answer(completion) or ""
        return 1.0 if parsed_answer.strip() == answer.strip() else 0.0
    
    # PII Count Reward Function
    def pii_count_reward(parser, completion, info) -> float:
        parsed_answer = parser.parse_answer(completion) or ""
        expected_count = info.get("pii_count")
        actual_count = parsed_answer.count("[PII]")
        return 1.0 if actual_count == expected_count else 0.0

    # ===== Rubric =====

    # Define Rubric
    rubric = vf.Rubric(
        parser=parser,
        funcs=[
            exact_match_reward,
            pii_count_reward,
            format_reward,
        ],
        weights=[1.0, 0.5, 0.1],
    )

    # ===== Environment =====

    # Define Environment
    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    # Return Environment
    return vf_env