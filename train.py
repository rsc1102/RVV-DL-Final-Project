import torch
import numpy as np
import random
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset , load_metric
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import TrainerState

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("gretelai/synthetic_text_to_sql")
dataset = dataset.filter(lambda example: example['sql_complexity'] in ['basic SQL','aggregation'])
dataset = dataset.filter(lambda example: example['sql_task_type'] == 'analytics and reporting')
temp_dataset = dataset["train"].train_test_split(test_size=0.1,seed=42)
dataset['train'] = temp_dataset['train']
dataset['validation'] = temp_dataset['test']

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer.add_tokens(['<sql_context>'])
model.resize_token_embeddings(len(tokenizer))

def preprocess_data(examples):
  input_text = examples['sql_prompt'] + " \n<sql_context>\n " + examples['sql_context']
  inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=512)
  inputs['labels'] = tokenizer(examples['sql'],padding="max_length",truncation=True, max_length=512)['input_ids']
  return inputs

tokenized_datasets = dataset.map(preprocess_data)

tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)

batch_size = 16
model_name = "t5-small"
model_dir = f"./{model_name}"
model_save_path = f"./{model_name}"

training_args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=10,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained(model_save_path, from_pt=True)
tokenizer.save_pretrained(model_save_path, from_pt=True)

