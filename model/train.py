from transformers import GPT2Config, GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

hf_tokenizer = PreTrainedTokenizerFast.from_pretrained("./my_tokenizer")

config = GPT2Config(
    vocab_size=len(hf_tokenizer),
    n_positions=512,
    n_ctx=512,
    n_embd=384,
    n_layer=6,
    n_head=6,
    pad_token_id=hf_tokenizer.pad_token_id
)

model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(hf_tokenizer))


dataset = load_dataset("text", data_files={"train": "../data/processed_tokens.txt"})
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


block_size = 512
stride = 256

def sliding_tokenizer(batch):
    all_input_ids = []
    all_attention_masks = []

    for text in batch["text"]:
        input_ids = hf_tokenizer(text)["input_ids"]
        input_ids = [x for x in input_ids if x != hf_tokenizer.pad_token_id]

        for i in range(0, len(input_ids) - block_size + 1, stride):
            chunk = input_ids[i:i + block_size]
            all_input_ids.append(chunk)
            all_attention_masks.append([1] * len(chunk))

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
    }


# Apply sliding tokenizer
tokenized_train = train_dataset.map(
    sliding_tokenizer,
    batched=True,
    remove_columns=["text"]
)

tokenized_eval = eval_dataset.map(
    sliding_tokenizer,
    batched=True,
    remove_columns=["text"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./model_checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    warmup_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,
    weight_decay=0.01,
    prediction_loss_only=True,
    fp16=True  # Enable if using a GPU with FP16 support
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./lol-gpt-medium")
hf_tokenizer.save_pretrained("./lol-gpt-medium")

with open("../data/train_tokens.txt", "w") as f:
    for ex in train_dataset:
        f.write(ex["text"] + "\n")

with open("../data/eval_tokens.txt", "w") as f:
    for ex in eval_dataset:
        f.write(ex["text"] + "\n")
