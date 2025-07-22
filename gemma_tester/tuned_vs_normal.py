import random
import csv
import re
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch

def generate_multiplication_prompts(num_prompts=200):
    prompts = []
    for _ in range(num_prompts):
        n1 = random.randint(100, 999)
        n2 = random.randint(100, 999)
        base_prompt = f"Answer the following just with the result of the multiplication. {n1:03d} x {n2:03d} = "
        formatted_prompt = f"<bos><start_of_turn>user\n{base_prompt}<end_of_turn>\n<start_of_turn>model\n"
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

def test_model_accuracy(model_name, prompts, output_file):
    print(f"Testing {model_name}...")

    if "furrutiav" in model_name.lower():
        base_model_name = "google/gemma-2-2b"
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="offload",
            low_cpu_mem_usage=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="offload",
            low_cpu_mem_usage=True
        )

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    results = []
    correct = 0

    for i, item in enumerate(prompts):
        if i % 20 == 0:
            print(f"Processing prompt {i}/{len(prompts)}")

        try:
            response = generator(
                item["prompt"],
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=generator.tokenizer.eos_token_id
            )

            generated_text = response[0]["generated_text"]
            answer_part = generated_text[len(item["prompt"]):].strip()

            predicted_answer, is_correct = extract_answer_from_response(answer_part, item["correct_answer"])

            if is_correct:
                correct += 1

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

    return {
        "model_name": model_name,
        "total_prompts": len(prompts),
        "correct_answers": correct,
        "accuracy": accuracy,
        "results": results
    }

if __name__ == "__main__":
    random.seed(42)
    prompts = generate_multiplication_prompts(200)

    models_to_test = [
        ("furrutiav/gemma-lora-2b", "furrutiav_gemma_lora_2b_performance.csv"),
        ("google/gemma-2-2b-it", "gemma_2_2b_it_performance.csv")
    ]

    for model_name, output_file in models_to_test:
        try:
            test_model_accuracy(model_name, prompts, output_file)
            print("-" * 50)
        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
            print("-" * 50)
