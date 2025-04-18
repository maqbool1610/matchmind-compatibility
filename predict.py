
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def predict(profile1, profile2):
    base_model_name = "microsoft/phi-1_5"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, "./lora_phi_output")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt = f"Profile A: {profile1}\nProfile B: {profile2}\nCompatibility:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.replace(prompt, "").strip()
