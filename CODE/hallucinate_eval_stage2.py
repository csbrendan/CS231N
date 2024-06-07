import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
import os
from datasets import load_dataset
import random
import json

DEVICE = "cuda:0"


os.environ['HF_HOME'] = "/home/bpm_azure_cs231n_key/huggingface_cache"
os.environ['TRANSFORMERS_CACHE'] = "/home/bpm_azure_cs231n_key/huggingface_cache"
os.environ["HF_TOKEN"] = "****"
hf_token = os.environ.get('HF_TOKEN')

processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)


model = Idefics2ForConditionalGeneration.from_pretrained(
    #"idefics2-8B-finetuned-stage2-full-train",
    "idefics2-8B-finetuned-stage2-ROCO-65k-train",
    torch_dtype=torch.float16,
).to(DEVICE)

dataset = load_dataset("flaviagiammarino/vqa-rad")
eval_dataset = dataset["test"]

def check_inference(model, processor, image, question, max_new_tokens=20):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Answer briefly."},
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True).to(DEVICE)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
    return generated_texts[0]

fake_questions = [
    "In the fantastical land where people walk upside down, where the sky is green and the grass is blue, what fantastical term defines the movement disorder characterized by the inane and nonsensical phenomenon of simultaneous contraction of agonist and antagonist muscles, inducing absurd postural changes and inexplicable movements?",
    "In the realm of improbability, where mermaids reign supreme with their uncanny possession of esophagi, by what unfathomable and inexplicable histological finding could the incomprehensible diagnosis of Barret\"s esophagus be ascertained in a Mermaid biopsy?",
    "Far beyond our known universe, in the mysterious and wondrous planet of Gorgons, where liver cancer claims countless lives of their benevolent creatures, which of the following preposterously absurd and nonsensical measures cannot serve as a surgical resection of poor prognostic factors for their liver cancer?",
    "Amidst the absurdity of a world entirely composed of candy and chocolates, which of the following incomprehensible and fantastical scenarios is not a reality?",
    "Regarding the laughably absurd concept of imaginary exercises with the incentive spirometer, which can only exist in the realm of fantasy and fiction, what unapologetically false and utterly ludicrous statement is somewhat true?"
]

results = []

for fake_question in fake_questions:
    # Randomly select an image from the test set
    random_index = random.randint(0, len(eval_dataset) - 1)
    random_example = eval_dataset[random_index]
    image = random_example['image']

    predicted_answer = check_inference(model, processor, image, fake_question, max_new_tokens=50).lower()

    result = {
        #'image': random_example['image'],
        'fake_question': fake_question,
        'predicted_answer': predicted_answer
    }
    results.append(result)

    #print(f"Image: {random_example['image']}")
    print(f"Fake Question: {fake_question}")
    print(f"Predicted Answer: {predicted_answer}")
    print()

# to JSON file
with open('eval_fake_questions_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)
