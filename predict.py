from cog import BasePredictor, Input
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re

class Predictor(BasePredictor):
    def setup(self):
        base_model_name = "microsoft/phi-1_5"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        base_model.resize_token_embeddings(len(self.tokenizer))
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model = PeftModel.from_pretrained(base_model, "./")
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def predict(self, profile1: str = Input(description="Profile A"),
                      profile2: str = Input(description="Profile B")) -> str:
        prompt = f"""
        You are a culturally aware compatibility expert with deep understanding of Indian family dynamics, fairness, dowry sensitivity, and emotional maturity.
        Evaluate the compatibility between two individuals based on their values and expectations. If there are any major conflicts in beliefs—especially around dowry, autonomy, respect, or fairness—call it out clearly.
        Respond with a single sentence conclusion only. Avoid generic advice or extra discussion.
        Profile A: {profile1}
        Profile B: {profile2}
        Compatibility:
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = generated.replace(prompt.strip(), "").strip()
        response = re.split(r'\n|Exercise|Answer|In a study|Based on this|Here is', response)[0]
        response = re.split(r'\.\s', response)[0].strip() + '.'

        return response
