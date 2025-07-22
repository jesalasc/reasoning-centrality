import random
from typing import Dict, List
import pandas as pd
import os

def generate_multiplication_prompts(num_samples: int = 100) -> List[Dict]:
    """Generate multiplication prompts with 3-digit numbers and their expected answers."""
    prompts = []
    for _ in range(num_samples):
        n1 = random.randint(0, 999)
        n2 = random.randint(0, 999)
        prompt = f"<bos><start_of_turn>user\nAnswer the following just with the result of the multiplication. {n1:03d} x {n2:03d} = <end_of_turn>\n<start_of_turn>model\n"
        expected_answer = n1 * n2
        prompts.append({
            'prompt': prompt,
            'expected_answer': expected_answer
        })
    return prompts

def save_multiplication_data(prompts: List[Dict], output_dir: str):
    """Save only the prompt and correct answer for each multiplication."""
    os.makedirs(output_dir, exist_ok=True)

    data = []
    for i, prompt_info in enumerate(prompts):
        data.append({
            'sample_id': i,
            'prompt': prompt_info['prompt'],
            'expected_answer': prompt_info['expected_answer']
        })

    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/multiplication_data.csv", index=False)

def main_func(num_samples: int = 200, output_dir: str = "gemma_multiplication/multiplication_data"):
    """Generate and save multiplication prompts with correct answers."""
    prompts = generate_multiplication_prompts(num_samples)
    save_multiplication_data(prompts, output_dir)
    print(f"Generated {num_samples} multiplication problems saved to {output_dir}/multiplication_data.csv")

if __name__ == "__main__":
    main_func(num_samples=50000)
