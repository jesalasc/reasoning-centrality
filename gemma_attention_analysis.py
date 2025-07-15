import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
import random
from typing import Dict, List
import gc

def load_model(model_name: str = "google/gemma-2-2b"):
    """Load the Gemma model and tokenizer."""
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
    print(f"Model loaded")
    return tokenizer, model

def generate_multiplication_prompts(num_samples: int = 100) -> List[str]:
    """Generate multiplication prompts with 3-digit numbers."""
    prompts = []
    for _ in range(num_samples):
        n1 = random.randint(0, 999)
        n2 = random.randint(0, 999)
        prompt = f"what is the value of {n1:03d} times {n2:03d}"
        prompts.append(prompt)
    return prompts

def extract_attention_and_embeddings(text: str, tokenizer, model) -> Dict:
    """Extract attention tensors and pre-attention embeddings for each token."""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        
    # Extract tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Extract attention weights (all layers, all heads)
    attentions = outputs.attentions  # tuple of (batch_size, num_heads, seq_len, seq_len)
    
    # Extract hidden states (embeddings before each attention layer)
    hidden_states = outputs.hidden_states  # tuple of (batch_size, seq_len, hidden_size)
    
    return {
        'tokens': tokens,
        'input_ids': inputs['input_ids'][0].cpu().numpy(),
        'attentions': [att[0].cpu().numpy() for att in attentions],  # Remove batch dimension
        'hidden_states': [hs[0].cpu().numpy() for hs in hidden_states],  # Remove batch dimension
        'prompt': text
    }

def save_attention_data(data: Dict, output_dir: str, sample_id: int):
    """Save attention and embedding data as CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokens and basic info
    token_df = pd.DataFrame({
        'token_id': range(len(data['tokens'])),
        'token': data['tokens'],
        'input_id': data['input_ids'],
        'prompt': [data['prompt']] * len(data['tokens'])
    })
    token_df.to_csv(f"{output_dir}/sample_{sample_id:03d}_tokens.csv", index=False)
    
    # Save attention weights for each layer
    for layer_idx, attention_layer in enumerate(data['attentions']):
        # attention_layer shape: (num_heads, seq_len, seq_len)
        num_heads, seq_len, _ = attention_layer.shape
        
        # Flatten attention matrix for CSV storage
        attention_records = []
        for head_idx in range(num_heads):
            for from_token in range(seq_len):
                for to_token in range(seq_len):
                    attention_records.append({
                        'layer': layer_idx,
                        'head': head_idx,
                        'from_token': from_token,
                        'to_token': to_token,
                        'from_token_text': data['tokens'][from_token],
                        'to_token_text': data['tokens'][to_token],
                        'attention_weight': attention_layer[head_idx, from_token, to_token]
                    })
        
        attention_df = pd.DataFrame(attention_records)
        attention_df.to_csv(f"{output_dir}/sample_{sample_id:03d}_attention_layer_{layer_idx:02d}.csv", index=False)
    
    # Save embeddings (hidden states) for each layer
    for layer_idx, hidden_state in enumerate(data['hidden_states']):
        # hidden_state shape: (seq_len, hidden_size)
        seq_len, hidden_size = hidden_state.shape
        
        # Create embedding records
        embedding_records = []
        for token_idx in range(seq_len):
            record = {
                'layer': layer_idx,
                'token_id': token_idx,
                'token': data['tokens'][token_idx],
            }
            # Add each embedding dimension as a separate column
            for dim_idx in range(hidden_size):
                record[f'embed_dim_{dim_idx:04d}'] = hidden_state[token_idx, dim_idx]
            
            embedding_records.append(record)
        
        embedding_df = pd.DataFrame(embedding_records)
        embedding_df.to_csv(f"{output_dir}/sample_{sample_id:03d}_embeddings_layer_{layer_idx:02d}.csv", index=False)

def analyze_multiplication_samples(num_samples: int = 10, output_dir: str = "gemma_analysis_data"):
    """Analyze multiple multiplication samples and save results."""
    tokenizer, model = load_model()
    prompts = generate_multiplication_prompts(num_samples)
    
    print(f"Analyzing {num_samples} multiplication samples...")
    for i, prompt in enumerate(prompts):
        print(f"Processing sample {i+1}/{num_samples}: {prompt}")
        
        try:
            data = extract_attention_and_embeddings(prompt, tokenizer, model)
            save_attention_data(data, output_dir, i)
            
            # Clear memory
            del data
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Analysis complete! Data saved to {output_dir}/")

if __name__ == "__main__":
    analyze_multiplication_samples(num_samples=20, output_dir="gemma_multiplication_analysis")