import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration, Blip2Model
from models.T5_model import T5ForMultimodalGeneration
from PIL import Image


class Agent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm, critic_lm, do_sample, temperature, max_new_tokens):
        super(Agent, self).__init__()

        print(f"### load policy lm: {policy_lm}")
        self.model = T5ForMultimodalGeneration.from_pretrained(policy_lm, torch_dtype=torch.bfloat16).to(device)
        
        print(f"### load critic && trajectory critic: {critic_lm}")
        self.critic = Qwen2VLForConditionalGeneration.from_pretrained(critic_lm, torch_dtype=torch.bfloat16).to(device)
        self.critic_processor = AutoProcessor.from_pretrained(critic_lm)

        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=0.5)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
    

    def prepare(self):
        self.model = self.accelerator.prepare(self.model)
        self.critic = self.accelerator.prepare(self.critic)


    def get_log_prob(self, text, image_features, target):
        image_features = image_features[..., -1408:]
        text_ids = self.tokenizer(text, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        target_ids = self.tokenizer(target, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        outputs = self.model(
            input_ids = text_ids["input_ids"],
            image_ids = image_features,
            attention_mask = text_ids["attention_mask"],
            labels = target_ids["input_ids"]
        )
        
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs, target_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        selected_prediction_probs = torch.clamp(selected_prediction_probs, min=0.001, max=0.99)
        
        return torch.log(selected_prediction_probs) * target_ids["attention_mask"]
    
    
    def get_action(self, texts, image_features):
        image_features = image_features[..., -1408:]
        
        with torch.no_grad():
            text_ids = self.tokenizer(texts, return_tensors='pt', padding=True, max_length=512, truncation=True).to(self.device)
            image_features = image_features.to(self.device)
            outputs = self.accelerator.unwrap_model(self.model).generate(
                **text_ids, image_ids=image_features,
                max_new_tokens=self.max_new_tokens, 
                do_sample=self.do_sample, 
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            ).cpu()
        
        raw_actions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for _ in range(3):
            raw_actions = [a[1:] if a.startswith('\n') else a for a in raw_actions]
        
        return raw_actions


class ImageFeatureExtractor:
    def __init__(self):
        self.model = Blip2Model.from_pretrained("./checkpoints/blip2-opt-2.7b").to("cuda")
        self.processor = AutoProcessor.from_pretrained("./checkpoints/blip2-opt-2.7b")

    def to_feat(self, image_path: str):
        with torch.no_grad():
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to("cuda")
            
            image_features = self.model.get_image_features(**inputs).pooler_output[0]
            image_features = image_features.detach().cpu()
            
        return image_features
