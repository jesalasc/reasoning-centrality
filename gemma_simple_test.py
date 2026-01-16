#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_gemma_model():
    """Load the Gemma-2-2B model and tokenizer."""
    model_name = "google/gemma-2-2b-it"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"
    )
    print("Model loaded successfully!")
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_tokens=50):
    """Generate a response from the model."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Input: {prompt}")
    print("Generating response...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the new tokens (the model's response)
    response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    return full_response, response

def main():
    # Load the model
    tokenizer, model = load_gemma_model()

    # Test prompt - you can change this to whatever you want
    test_prompt = f"<bos><start_of_turn>user\nAnswer the following just with the result of the multiplication. 352 x 236 = <end_of_turn>\n<start_of_turn>model\n"

    # Generate response
    full_response, model_response = generate_response(test_prompt, tokenizer, model)

    # Display results
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(f"Prompt: {test_prompt}")
    print(f"Model Response: {model_response}")
    print(f"Full Text: {full_response}")
    print("="*50)

if __name__ == "__main__":
    main()
