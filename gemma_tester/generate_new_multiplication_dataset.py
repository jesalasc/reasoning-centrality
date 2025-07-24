import random
import csv
import re

def extract_existing_pairs(csv_file_path):
    """Extract all existing multiplication pairs from the CSV file"""
    seen_pairs = set()

    with open(csv_file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) >= 2:
                prompt = row[1]
                # Extract numbers from prompt using regex
                match = re.search(r'(\d{3}) x (\d{3})', prompt)
                if match:
                    n1, n2 = int(match.group(1)), int(match.group(2))
                    # Add both (n1, n2) and (n2, n1) to avoid commutative duplicates
                    seen_pairs.add((n1, n2))
                    seen_pairs.add((n2, n1))

    return seen_pairs

def generate_new_multiplication_dataset(num_samples=10000, existing_csv_path=None):
    """Generate new multiplication dataset avoiding existing pairs"""

    # Load existing pairs if provided
    seen_pairs = set()
    if existing_csv_path:
        print(f"Loading existing pairs from {existing_csv_path}...")
        seen_pairs = extract_existing_pairs(existing_csv_path)
        print(f"Found {len(seen_pairs)} existing pairs to avoid")

    samples = []
    attempts = 0
    max_attempts = num_samples * 10  # Prevent infinite loop

    print(f"Generating {num_samples} new multiplication samples...")

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1

        # Generate random 3-digit numbers
        n1 = random.randint(000, 999)
        n2 = random.randint(000, 999)

        # Check if this pair already exists
        if (n1, n2) not in seen_pairs and (n2, n1) not in seen_pairs:
            # Add to seen pairs to avoid duplicates within new dataset
            seen_pairs.add((n1, n2))
            seen_pairs.add((n2, n1))

            # Create prompt in same format as existing data
            prompt = f'<start_of_turn>user\nAnswer the following just with the result of the multiplication. {n1:03d} x {n2:03d} = <end_of_turn>\n<start_of_turn>model\n'
            expected_answer = n1 * n2

            samples.append({
                'sample_id': len(samples),
                'prompt': prompt,
                'expected_answer': expected_answer,
                'n1': n1,
                'n2': n2
            })

            if len(samples) % 1000 == 0:
                print(f"Generated {len(samples)} samples...")

    if attempts >= max_attempts:
        print(f"Warning: Reached maximum attempts. Generated {len(samples)} samples instead of {num_samples}")

    return samples

def save_dataset_to_csv(samples, output_path):
    """Save the dataset to CSV file"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'prompt', 'expected_answer'])

        for sample in samples:
            writer.writerow([
                sample['sample_id'],
                sample['prompt'],
                sample['expected_answer']
            ])

    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Path to existing multiplication data
    existing_csv_path = "/Users/canonrobins/Documents/GitHub/reasoning-centrality/gemma_multiplication/multiplication_data/multiplication_data.csv"

    # Generate new dataset
    new_samples = generate_new_multiplication_dataset(
        num_samples=10000,
        existing_csv_path=existing_csv_path
    )

    # Save to CSV
    output_path = "/Users/canonrobins/Documents/GitHub/reasoning-centrality/gemma_tester/new_multiplication_data.csv"
    save_dataset_to_csv(new_samples, output_path)

    print(f"\nDataset generation complete!")
    print(f"Generated {len(new_samples)} unique multiplication samples")
    print(f"Saved to: {output_path}")
