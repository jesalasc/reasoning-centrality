import random
import csv
import re
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel, PeftConfig
import torch
import gc

def force_cleanup():
    """Quick cleanup between models for MPS"""
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def load_multiplication_prompts_from_csv(csv_path, num_samples=None):
    """Load multiplication prompts from the new dataset CSV"""
    prompts = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) >= 3:
                sample_id = int(row[0])
                prompt = row[1]
                expected_answer = int(row[2])

                # Extract n1 and n2 from prompt
                match = re.search(r'(\d{3}) x (\d{3})', prompt)
                if match:
                    n1, n2 = int(match.group(1)), int(match.group(2))

                    # Convert prompt format to match current expectations
                    formatted_prompt = prompt.replace('<bos>', '').replace('<start_of_turn>user\n', '<start_of_turn>user\n').replace('<end_of_turn>\n<start_of_turn>model\n', '<end_of_turn>\n<start_of_turn>model\n')

                    prompts.append({
                        "prompt": formatted_prompt,
                        "n1": n1,
                        "n2": n2,
                        "correct_answer": expected_answer
                    })

    # Shuffle and limit to requested number of samples
    random.shuffle(prompts)
    if num_samples and num_samples < len(prompts):
        prompts = prompts[:num_samples]

    return prompts

def generate_multiplication_prompts(num_prompts=200):
    """Legacy function for backward compatibility"""
    prompts = []
    for _ in range(num_prompts):
        n1 = random.randint(100, 999)
        n2 = random.randint(100, 999)
        base_prompt = f"Answer the following just with the result of the multiplication. {n1:03d} x {n2:03d} = "
        formatted_prompt = f"<start_of_turn>user\n{base_prompt}<end_of_turn>\n<start_of_turn>model\n"
        correct_answer = n1 * n2
        prompts.append({
            "prompt": formatted_prompt,
            "n1": n1,
            "n2": n2,
            "correct_answer": correct_answer
        })
    return prompts

def extract_answer_from_response(response_text, correct_answer):
    response_text = response_text.strip()

    # Limit to first 15 tokens
    tokens = response_text.split()[:15]
    response_segment = ' '.join(tokens)

    correct_str = str(correct_answer)
    correct_with_commas = f"{correct_answer:,}"

    # Check if the entire response (or response with whitespace) is just a number
    if response_segment.replace(',', '').isdigit():
        try:
            num = int(response_segment.replace(',', ''))
            return num, num == correct_answer
        except ValueError:
            pass

    # Check if correct answer appears anywhere in the response segment
    patterns = [
        rf'(?<!\d){re.escape(correct_str)}(?!\d)',
        rf'(?<!\d){re.escape(correct_with_commas)}(?!\d)',
    ]

    for pattern in patterns:
        if re.search(pattern, response_segment):
            return correct_answer, True

    # Find all numbers in the response segment and check if any match
    numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*\b', response_segment)
    for num_str in numbers:
        try:
            num = int(num_str.replace(',', ''))
            if num == correct_answer:
                return num, True
        except ValueError:
            continue

    # Return first found number if any, marked as incorrect
    if numbers:
        try:
            return int(numbers[0].replace(',', '')), False
        except ValueError:
            return None, False

    return None, False

def calculate_digit_metrics(correct_answer, predicted_answer):
    """Calculate digit-level accuracy metrics"""
    if predicted_answer is None:
        return float('inf'), float('inf')

    correct_str = str(correct_answer)
    predicted_str = str(predicted_answer)

    # Pad shorter number with leading zeros for comparison
    max_len = max(len(correct_str), len(predicted_str))
    correct_padded = correct_str.zfill(max_len)
    predicted_padded = predicted_str.zfill(max_len)

    incorrect_digits = 0
    digit_diff_sum = 0

    for i in range(max_len):
        correct_digit = int(correct_padded[i])
        predicted_digit = int(predicted_padded[i])

        if correct_digit != predicted_digit:
            incorrect_digits += 1

        digit_diff_sum += abs(correct_digit - predicted_digit)

    return incorrect_digits, digit_diff_sum

def test_model_accuracy(model_name, prompts, output_file):
    print(f"Testing {model_name}...")

    if "crobins" in model_name.lower():
        # Method 1: Try AutoPeftModelForCausalLM first
        try:
            # Create offload directory if it doesn't exist
            offload_dir = "./offload"
            os.makedirs(offload_dir, exist_ok=True)

            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder=offload_dir,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"AutoPeftModelForCausalLM failed: {e}")
            # Method 2: Fallback to manual loading
            try:
                config = PeftConfig.from_pretrained(model_name)
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    offload_folder=offload_dir,
                    low_cpu_mem_usage=True
                )
                model = PeftModel.from_pretrained(base_model, model_name)
                tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            except Exception as e2:
                print(f"Manual loading also failed: {e2}")
                # Method 3: Last resort - load base model first
                base_model = AutoModelForCausalLM.from_pretrained(
                    "google/gemma-2-2b-it",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    offload_folder=offload_dir,
                    low_cpu_mem_usage=True
                )
                model = PeftModel.from_pretrained(base_model, model_name)
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    else:
        # Create offload directory if it doesn't exist
        offload_dir = "./offload"
        os.makedirs(offload_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_dir,
            low_cpu_mem_usage=True
        )

    # Check what token ID 2 actually represents
    # print("Method 1 - String:", tokenizer.encode("<bos>"))
    # print("Method 2 - bos_token:", tokenizer.encode(tokenizer.bos_token))
    # print("Method 3 - Direct ID:", [tokenizer.bos_token_id])

    # Check if you should be using the tokenizer's built-in method
    # print("Built-in BOS:", repr(tokenizer.bos_token))

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    results = []
    correct = 0
    total_incorrect_digits = 0
    total_digit_diff_sum = 0
    valid_digit_comparisons = 0

    for i, item in enumerate(prompts):
        if i % 20 == 0:
            print(f"Processing prompt {i}/{len(prompts)}")

        try:
            response = generator(
                item["prompt"],
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=generator.tokenizer.eos_token_id,
                early_stopping=True,
                num_return_sequences=1,
                num_beams=5
            )

            generated_text = response[0]["generated_text"]
            answer_part = generated_text[len(item["prompt"]):].strip()

            predicted_answer, is_correct = extract_answer_from_response(answer_part, item["correct_answer"])

            if is_correct:
                correct += 1

            # Calculate digit-level metrics
            incorrect_digits, digit_diff_sum = calculate_digit_metrics(item["correct_answer"], predicted_answer)

            if incorrect_digits != float('inf'):
                total_incorrect_digits += incorrect_digits
                total_digit_diff_sum += digit_diff_sum
                valid_digit_comparisons += 1

            results.append({
                "prompt": item["prompt"],
                "n1": item["n1"],
                "n2": item["n2"],
                "correct_answer": item["correct_answer"],
                "raw_response": answer_part,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct
            })

        except Exception as e:
            print(f"Error processing prompt {i}: {e}")
            results.append({
                "prompt": item["prompt"],
                "n1": item["n1"],
                "n2": item["n2"],
                "correct_answer": item["correct_answer"],
                "raw_response": f"Error: {str(e)}",
                "predicted_answer": None,
                "is_correct": False
            })

    accuracy = correct / len(prompts)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'total_prompts', 'correct_answers', 'accuracy'])
        writer.writerow([model_name, len(prompts), correct, accuracy])
        writer.writerow([])
        writer.writerow(['prompt', 'n1', 'n2', 'correct_answer', 'raw_response', 'predicted_answer', 'is_correct'])
        for result in results:
            writer.writerow([
                result['prompt'],
                result['n1'],
                result['n2'],
                result['correct_answer'],
                result['raw_response'],
                result['predicted_answer'],
                result['is_correct']
            ])

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(prompts)})")
    print(f"Results saved to: {output_path}")

    # Calculate average and median difference between true and predicted answers
    differences = []
    total_diff = 0
    valid_predictions = 0

    for result in results:
        if result["predicted_answer"] is not None:
            diff = abs(result["correct_answer"] - result["predicted_answer"])
            differences.append(diff)
            total_diff += diff
            valid_predictions += 1

    avg_difference = total_diff / valid_predictions if valid_predictions > 0 else float('inf')

    # Calculate median difference
    if differences:
        differences.sort()
        n = len(differences)
        if n % 2 == 0:
            median_difference = (differences[n//2 - 1] + differences[n//2]) / 2
        else:
            median_difference = differences[n//2]
    else:
        median_difference = float('inf')

    # Calculate digit-level averages
    avg_incorrect_digits = total_incorrect_digits / valid_digit_comparisons if valid_digit_comparisons > 0 else float('inf')
    avg_digit_diff_sum = total_digit_diff_sum / valid_digit_comparisons if valid_digit_comparisons > 0 else float('inf')

    return {
        "model_name": model_name,
        "total_prompts": len(prompts),
        "correct_answers": correct,
        "accuracy": accuracy,
        "avg_difference": avg_difference,
        "median_difference": median_difference,
        "avg_incorrect_digits": avg_incorrect_digits,
        "avg_digit_diff_sum": avg_digit_diff_sum,
        "results": results
    }

def create_comparison_csv(all_results, output_file="model_comparison.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model Name', 'Total Prompts', 'Correct Answers', 'Accuracy', 'Average Difference', 'Median Difference', 'Avg Incorrect Digits', 'Avg Digit Diff Sum'])

        for result in all_results:
            writer.writerow([
                result['model_name'],
                result['total_prompts'],
                result['correct_answers'],
                f"{result['accuracy']:.4f}",
                f"{result['avg_difference']:.2f}" if result['avg_difference'] != float('inf') else "N/A",
                f"{result['median_difference']:.2f}" if result['median_difference'] != float('inf') else "N/A",
                f"{result['avg_incorrect_digits']:.2f}" if result['avg_incorrect_digits'] != float('inf') else "N/A",
                f"{result['avg_digit_diff_sum']:.2f}" if result['avg_digit_diff_sum'] != float('inf') else "N/A"
            ])

    print(f"Comparison results saved to: {output_path}")

if __name__ == "__main__":
    # Configuration parameters
    NUM_SAMPLES = 1000  # Number of samples to use from the dataset
    USE_NEW_DATASET = True  # Set to False to use legacy random generation

    random.seed(33)

    if USE_NEW_DATASET:
        # Load prompts from the new dataset
        csv_path = "new_multiplication_data.csv"
        prompts = load_multiplication_prompts_from_csv(csv_path, NUM_SAMPLES)
        print(f"Loaded {len(prompts)} prompts from new dataset: {csv_path}")
    else:
        # Use legacy random generation
        prompts = generate_multiplication_prompts(NUM_SAMPLES)
        print(f"Generated {len(prompts)} random prompts")

    models_to_test = [
        ("google/gemma-2-2b-it", "gemma_2_2b_it_performance.csv"),
        ("crobins/gemma_2_2b_lora", "crobins_gemma_2_2b_lora_performance.csv")
    ]

    all_results = []

    for model_name, output_file in models_to_test:
        try:
            force_cleanup()
            result = test_model_accuracy(model_name, prompts, output_file)
            all_results.append(result)
            print("-" * 50)
        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
            force_cleanup()
            print("-" * 50)

    # Create comparison CSV
    if all_results:
        create_comparison_csv(all_results)
