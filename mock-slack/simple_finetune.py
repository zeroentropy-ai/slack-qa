#!/usr/bin/env python3
"""
Simple fine-tuning script without wandb dependencies.
Minimal version focused on just training the model.
"""
import json
import os
import torch
from typing import Dict, List, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Model configuration  
MODEL_NAME = "Qwen/Qwen2.5-8B-Instruct"
OUTPUT_DIR = "./qwen-search-query-model"

def load_training_data(file_path: str = "finetuning_data.json") -> List[Dict[str, Any]]:
    """Load training data from finetuning_data.json"""
    with open(file_path, 'r') as f:
        return json.load(f)

def format_training_example(example: Dict[str, Any]) -> str:
    """Format a training example for instruction tuning"""
    question = example['question']
    search_queries = json.dumps(example['successful_search_queries'])
    
    # Simple format without complex chat template
    conversation = f"Question: {question}\nSearch queries: {search_queries}"
    
    return conversation

def prepare_dataset(training_data: List[Dict[str, Any]], tokenizer) -> Dataset:
    """Prepare dataset for training"""
    
    # Format all examples
    formatted_examples = []
    for example in training_data:
        formatted_text = format_training_example(example)
        formatted_examples.append({"text": formatted_text})
    
    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=256,  # Shorter for simpler format
            return_overflowing_tokens=False,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Create dataset
    dataset = Dataset.from_list(formatted_examples)
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return dataset

def setup_model_and_tokenizer():
    """Setup model and tokenizer with LoRA"""
    
    print(f"Loading model and tokenizer: {MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model directly to GPU for H100 with Flash Attention
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try Flash Attention 2 first, fallback to regular attention
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": device},
            attn_implementation="flash_attention_2",
            use_cache=False
        )
        print("✅ Using Flash Attention 2")
    except Exception as e:
        print(f"⚠️  Flash Attention 2 not available: {e}")
        print("Using standard attention implementation")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": device},
            use_cache=False
        )
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Smaller rank for faster training
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Fewer modules for speed
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset: Dataset):
    """Fine-tune the model with minimal configuration"""
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Flash Attention allows larger batches
        gradient_accumulation_steps=2,  # Effective batch size of 16
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        bf16=True,
        optim="adamw_torch",
        report_to=[],
        dataloader_num_workers=4,  # More workers with Flash Attention efficiency
        max_grad_norm=1.0,
        # Flash Attention optimizations
        dataloader_persistent_workers=True,
        group_by_length=True,  # Group similar length sequences for efficiency
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    return trainer

def test_model(model, tokenizer):
    """Quick test of the model"""
    
    print("\n🧪 Testing model:")
    
    test_questions = [
        "How do I fix GPU memory errors?",
        "Why is my billing high?",
        "How to deploy to production?"
    ]
    
    model.eval()
    
    for question in test_questions:
        prompt = f"Question: {question}\nSearch queries:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = response[len(prompt):].strip()
        
        print(f"\nQ: {question}")
        print(f"A: {generated_part}")

def main():
    print("🚀 Simple Qwen Fine-tuning")
    print("=" * 40)
    
    training_file = "finetuning_data.json"
    
    if not os.path.exists(training_file):
        print(f"❌ Training file {training_file} not found!")
        return
    
    try:
        # Load data
        print(f"Loading training data...")
        training_data = load_training_data(training_file)
        print(f"Loaded {len(training_data)} examples")
        
        # Setup model
        model, tokenizer = setup_model_and_tokenizer()
        
        # Prepare dataset
        train_dataset = prepare_dataset(training_data, tokenizer)
        print(f"Prepared dataset with {len(train_dataset)} examples")
        
        # Train
        train_model(model, tokenizer, train_dataset)
        
        # Test
        test_model(model, tokenizer)
        
        print(f"\n✅ Training completed! Model saved to {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()