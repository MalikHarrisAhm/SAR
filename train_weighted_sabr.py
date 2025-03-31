import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import os
import warnings
from SaBR import TokenWeightedTrainer, TokenWeightConfig, EnhancedMetaWeightNetwork

# Set environment and memory configurations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cuda.max_split_size_mb = 128

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", message="NCCL OOB")

# Initialize Accelerate's PartialState for distributed training
dist_state = PartialState()
if dist_state.is_main_process:
    print(f"Using {dist_state.num_processes} GPUs")

# Define model name and local storage path
model_name = 'unsloth/gemma-2-27b-it'
# model_name = 'gemma'
local_path = './downloaded_model'

# Only the main process downloads and saves the model
if dist_state.is_main_process:
    print("Main process handling model setup...")

    # Download and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)
    tokenizer.save_pretrained(local_path)

    # Download and save model
    print("Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation='eager',
        trust_remote_code=True
    )
    model.save_pretrained(local_path)
    print("Model saved locally")

# Synchronize all processes before proceeding
if dist_state.num_processes > 1:
    dist_state.wait_for_everyone()

# Now all processes can load the saved model
if not dist_state.is_main_process:
    print(f"Process {dist_state.process_index} loading saved model...")

tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    local_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    attn_implementation='eager',
    trust_remote_code=True,
    device_map={"": dist_state.local_process_index}
)

# Store hidden size before any wrapping
model_hidden_size = base_model.config.hidden_size

# Enable input gradients
base_model.enable_input_require_grads()

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=12,
    lora_alpha=128,
    lora_dropout=0.2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "ff_proj"],
    bias="none"
)

# Apply LoRA configuration and enable gradient checkpointing BEFORE DDP wrapping
model = get_peft_model(base_model, peft_config)
model.gradient_checkpointing_enable()

# Wrap the model in DistributedDataParallel if needed
# if dist_state.num_processes > 1:
#     model = torch.nn.parallel.DistributedDataParallel(
#         model,
#         device_ids=[dist_state.local_process_index],
#         output_device=dist_state.local_process_index,
#         find_unused_parameters=False
#     )

# Load training and validation datasets
train_dataset = load_dataset('json', data_files='balanced_reviews_3000_token.jsonl', split='train')
val_dataset = load_dataset('json', data_files='balanced_reviews_validation.jsonl', split='train')

# Shard datasets for distributed training
if dist_state.num_processes > 1:
    train_dataset = train_dataset.shard(
        num_shards=dist_state.num_processes,
        index=dist_state.process_index
    )
    val_dataset = val_dataset.shard(
        num_shards=dist_state.num_processes,
        index=dist_state.process_index
    )

# Define preprocessing function
def preprocess(example):
    if dist_state.is_main_process and example.get('__index__', 0) % 1000 == 0:
        print(f"GPU {dist_state.process_index}: Processing example {example.get('__index__', 0)}")

    max_prompt_length = 3000
    max_response_length = 3000

    prompt = f"Instruction: {example['instruction']}\nResponse:"
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_prompt_length)['input_ids']
    response_ids = tokenizer(example['response'], truncation=True, max_length=max_response_length)['input_ids']
    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    labels = input_ids.copy()
    return {'input_ids': input_ids, 'labels': labels}

# Preprocess datasets
train_dataset = train_dataset.map(
    preprocess,
    remove_columns=train_dataset.column_names,
    batch_size=1000,
    num_proc=4
)
val_dataset = val_dataset.map(
    preprocess,
    remove_columns=val_dataset.column_names,
    batch_size=1000,
    num_proc=4
)

# Set dataset formats
train_dataset.set_format('torch')
val_dataset.set_format('torch')

# Define token weights
token_weights = {
    # "Include": 10.0,
    "Exclude": 20.0
    # "include": 80.0,
    # "exclude": 80.0,
    # "included": 80.0,
    # "excluded": 80.0
}

# Configure training arguments
# Configure training arguments
training_args = TrainingArguments(
    output_dir='./finetuned_model',
    overwrite_output_dir=True,
    num_train_epochs=60,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=30,
    bf16=True,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=5,
    save_total_limit=100,
    save_safetensors=True,
    logging_steps=1,
    evaluation_strategy="steps",
    eval_steps=5,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    learning_rate=5e-5,  # Fixed at 5e-5 as requested
    lr_scheduler_type="constant",
    # warmup_steps=1,
    weight_decay=0.0,
    ddp_backend="nccl",
    ddp_find_unused_parameters=False,
    local_rank=dist_state.local_process_index,
    remove_unused_columns=False,
    report_to="none"
)

# Initialize token weight config
weight_config = TokenWeightConfig(
    hidden_dim=256,
    memory_size=10000,
    min_weight=0.1,
    max_weight=5.0,
    importance_lr=0.01,
    token_weights=token_weights,
    default_weight=1.0,
    gradient_clip=1.0,
    warmup_steps=2,
    update_interval=10
)

# Initialize custom EnhancedMetaWeightNetwork with explicit hidden size
meta_network = EnhancedMetaWeightNetwork(
    hidden_size=model_hidden_size,
    config=weight_config
).to(training_args.device)

# Initialize the trainer
trainer = TokenWeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt"),
    processing_class=tokenizer,
    meta_network=meta_network
)

try:
    # Train the model
    if os.path.exists("./finetuned_model/checkpoint-0"):
        trainer.train(resume_from_checkpoint="./finetuned_model/checkpoint-0")
    else:
        trainer.train()

    # Save the final model
    if dist_state.is_main_process:
        model.save_pretrained('./finetuned_model')
        print("Training complete and model saved.")

finally:
    # Clean up distributed training
    if dist.is_initialized():
        dist.destroy_process_group()