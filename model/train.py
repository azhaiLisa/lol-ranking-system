from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast

hf_tokenizer = PreTrainedTokenizerFast.from_pretrained("./my_tokenizer")

config = GPT2Config(
    vocab_size=len(hf_tokenizer),  # match your custom tokenizer
    n_positions=512,               # max token sequence length
    n_ctx=512,
    n_embd=384,                    # hidden size
    n_layer=6,                     # 6 transformer layers
    n_head=6,                      # 6 attention heads
    pad_token_id=hf_tokenizer.pad_token_id
)

model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(hf_tokenizer))  # match tokenizer vocab


dataset = load_dataset("text", data_files={"train": "../data/processed_tokens.txt"})

def tokenize_function(example):
    return hf_tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


training_args = TrainingArguments(
    output_dir="./model_checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    warmup_steps=50,
    learning_rate=5e-4,
    weight_decay=0.01,
    prediction_loss_only=True,
    fp16=False  # Enable if you’re on GPU with mixed precision support (e.g. Colab)
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer, mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./lol-gpt-medium")
hf_tokenizer.save_pretrained("./lol-gpt-medium")
