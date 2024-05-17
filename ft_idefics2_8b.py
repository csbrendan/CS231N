import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
import os

DEVICE = "cuda:0"

os.environ['HF_HOME'] = "/home/bpm_azure_cs231n_key/huggingface_cache"
os.environ['TRANSFORMERS_CACHE'] = "/home/bpm_azure_cs231n_key/huggingface_cache"
os.environ["HF_TOKEN"] = "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"
hf_token = os.environ.get('HF_TOKEN')

processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
    init_lora_weights="gaussian"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)
model.add_adapter(lora_config)
model.enable_adapters()


from datasets import load_dataset

#my full
#train_dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")
#eval_dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")

#my small test
dataset = load_dataset("flaviagiammarino/vqa-rad")
train_dataset = dataset["train"].select(range(1000))
eval_dataset = dataset["test"].select(range(100))

train_dataset[10]
train_dataset[10]["image"]
import random

class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            question = example["question"]
            answer = example["answer"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch


data_collator = MyDataCollator(processor)

output_dir = "/home/bpm_azure_cs231n_key/idefics_outputs"
os.makedirs(output_dir, exist_ok=True)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate = 1e-4,
    weight_decay=0.01,
    logging_steps=25,
    output_dir = "/home/bpm_azure_cs231n_key/idefics_outputs",
    save_strategy = "steps",
    save_steps = 25,
    save_total_limit = 1,
    fp16 = True,
    remove_unused_columns=False,
    report_to="none"
)
trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
example = eval_dataset[5]
example
example["image"]
model.eval()

image = example["image"]
query = example["question"]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Answer briefly."},
            {"type": "image"},
            {"type": "text", "text": example["question"]}
        ]
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True)
generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
print(generated_texts)

# save ft model
model.save_pretrained("idefics2-8B-finetuned")

torch.cuda.empty_cache()
