import random
from typing import Dict, List
import pandas as pd
import os

def generate_multiplication_prompts(num_samples: int = 100) -> List[Dict]:
    """Generate multiplication prompts with balanced complexity combinations and no duplicates."""
    prompts = []
    used_pairs = set()  # Track used (n1, n2) pairs to prevent duplicates
    
    # Define complexity categories with their ranges
    complexity_categories = [
        ("1x1", (1, 9), (1, 9)),      # 1-digit x 1-digit
        ("1x2", (1, 9), (10, 99)),    # 1-digit x 2-digit  
        ("1x3", (1, 9), (100, 999)),  # 1-digit x 3-digit
        ("2x2", (10, 99), (10, 99)),  # 2-digit x 2-digit
        ("2x3", (10, 99), (100, 999)), # 2-digit x 3-digit
        ("3x3", (100, 999), (100, 999)) # 3-digit x 3-digit
    ]
    
    # Calculate initial samples per category (ensure equal distribution)
    samples_per_category = num_samples // len(complexity_categories)
    remaining_samples = num_samples % len(complexity_categories)
    
    # First pass: determine available pairs for each category and adjust allocation
    category_info = []
    total_shortage = 0
    
    for i, (category_name, (min1, max1), (min2, max2)) in enumerate(complexity_categories):
        # Add one extra sample to first categories if there's a remainder
        requested_samples = samples_per_category + (1 if i < remaining_samples else 0)
        
        # Calculate total possible unique pairs for this category
        total_possible = (max1 - min1 + 1) * (max2 - min2 + 1)
        available_samples = min(requested_samples, total_possible)
        shortage = requested_samples - available_samples
        
        category_info.append({
            'name': category_name,
            'range': (min1, max1, min2, max2),
            'requested': requested_samples,
            'available': available_samples,
            'shortage': shortage,
            'total_possible': total_possible
        })
        
        total_shortage += shortage
    
    # Redistribute shortage to categories that can accommodate more samples
    if total_shortage > 0:
        # Find categories that can take extra samples
        expandable_categories = [cat for cat in category_info if cat['available'] < cat['total_possible']]
        
        if expandable_categories:
            # Distribute shortage equally among expandable categories
            extra_per_category = total_shortage // len(expandable_categories)
            extra_remainder = total_shortage % len(expandable_categories)
            
            for j, cat in enumerate(expandable_categories):
                extra_samples = extra_per_category + (1 if j < extra_remainder else 0)
                max_additional = cat['total_possible'] - cat['available']
                actual_extra = min(extra_samples, max_additional)
                cat['available'] += actual_extra
                total_shortage -= actual_extra
                
                if total_shortage == 0:
                    break
    
    # Second pass: generate the actual samples
    for cat in category_info:
        category_name = cat['name']
        min1, max1, min2, max2 = cat['range']
        category_samples = cat['available']
        
        if category_samples == 0:
            continue
            
        # Generate all possible pairs for this category
        possible_pairs = []
        for n1 in range(min1, max1 + 1):
            for n2 in range(min2, max2 + 1):
                pair = (n1, n2)
                if pair not in used_pairs:
                    possible_pairs.append(pair)
        
        # Randomly sample without replacement
        selected_pairs = random.sample(possible_pairs, min(category_samples, len(possible_pairs)))
        
        for n1, n2 in selected_pairs:
            used_pairs.add((n1, n2))
            prompt = f"<start_of_turn>user\nAnswer the following just with the result of the multiplication. {n1:03d} x {n2:03d} = <end_of_turn>\n<start_of_turn>model\n"
            expected_answer = n1 * n2
            prompts.append({
                'prompt': prompt,
                'expected_answer': expected_answer,
                'complexity': category_name,
                'n1': n1,
                'n2': n2
            })
    
    print(f"Generated {len(prompts)} samples (requested {num_samples})")
    for cat in category_info:
        print(f"  {cat['name']}: {cat['available']} samples (requested {cat['requested']}, max possible {cat['total_possible']})")
    
    # Shuffle to randomize order
    random.shuffle(prompts)
    return prompts

def save_multiplication_data(prompts: List[Dict], output_dir: str):
    """Save only the prompt and correct answer for each multiplication."""
    os.makedirs(output_dir, exist_ok=True)

    data = []
    for i, prompt_info in enumerate(prompts):
        data.append({
            'sample_id': i,
            'prompt': prompt_info['prompt'],
            'expected_answer': prompt_info['expected_answer'],
            'complexity': prompt_info['complexity'],
            'n1': prompt_info['n1'],
            'n2': prompt_info['n2']
        })

    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/multiplication_data.csv", index=False)

def main_func(num_samples: int = 200, output_dir: str = "multiplication_data"):
    """Generate and save multiplication prompts with correct answers."""
    prompts = generate_multiplication_prompts(num_samples)
    save_multiplication_data(prompts, output_dir)
    print(f"Generated {num_samples} multiplication problems saved to {output_dir}/multiplication_data.csv")

if __name__ == "__main__":
    main_func(num_samples=50000)
