import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
import os
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import json

DEVICE = "cuda:0"

# ENV vars
os.environ['HF_HOME'] = "/home/bpm_azure_cs231n_key/huggingface_cache"
os.environ['TRANSFORMERS_CACHE'] = "/home/bpm_azure_cs231n_key/huggingface_cache"
os.environ["HF_TOKEN"] = "hf_***************"
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
    quantization_config=bnb_config,
)
model.add_adapter(lora_config)
model.enable_adapters()

# EVAL on VQA-RAD test set
dataset = load_dataset("flaviagiammarino/path-vqa")
eval_dataset = dataset["test"].select(range(400)) 

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

def check_accuracy(model, processor, dataset, num_samples=20):
    exact_match_correct = 0
    f1_scores = []
    bleu_scores = []
    results = []

    for i in range(num_samples):
        example = dataset[i]
        image = example['image']
        question = example['question']
        true_answer = example['answer'].lower()
        predicted_answer = check_inference(model, processor, image, question, max_new_tokens=20).lower()

        # Exact Match Accuracy 
        if true_answer in predicted_answer:
            exact_match_correct += 1

        answer_start = predicted_answer.find("answer:")
        if answer_start != -1:
            answer_end = predicted_answer.find("question:", answer_start)
            if answer_end == -1:
                answer_end = len(predicted_answer)
            predicted_answer = predicted_answer[answer_start + len("answer:"):answer_end].strip()

        # Remove any "assistant:" text from the predicted answer
        predicted_answer = predicted_answer.replace("assistant:", "").strip()

        # Token-based F1 Score
        true_tokens = word_tokenize(true_answer)
        pred_tokens = word_tokenize(predicted_answer)
        common_tokens = set(true_tokens) & set(pred_tokens)
        precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = len(common_tokens) / len(true_tokens) if len(true_tokens) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)

        # BLEU Score 
        bleu_score = sentence_bleu([true_tokens], pred_tokens, weights=(1, 0, 0, 0))
        bleu_scores.append(bleu_score)

        result = {
            'question': question,
            'true_answer': true_answer,
            'predicted_answer': predicted_answer,
            'exact_match': true_answer in predicted_answer,
            'f1_score': f1_score,
            'bleu_score': bleu_score
        }
        results.append(result)

        print(f"Question: {question}")
        print(f"True Answer: {true_answer}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Exact Match: {'Yes' if true_answer in predicted_answer else 'No'}")
        print(f"F1 Score: {f1_score:.2f}")
        print(f"BLEU Score: {bleu_score:.2f}")
        print()

    exact_match_accuracy = exact_match_correct / num_samples
    avg_f1_score = sum(f1_scores) / len(f1_scores)
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    print(f"Exact Match Accuracy: {exact_match_accuracy:.2f}")
    print(f"Average F1 Score: {avg_f1_score:.2f}")
    print(f"Average BLEU Score: {avg_bleu_score:.2f}")

    return results

results = check_accuracy(model, processor, eval_dataset, num_samples=400)

# to JSON file
with open('baseline_path_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)