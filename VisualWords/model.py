# model.py

from typing import List
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class ImageCaptionGenerator:
    def __init__(self, top_k: int = 3, use_fp16: bool = True, use_compile: bool = True, model_name: str = "Salesforce/blip-image-captioning-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1️⃣ Load processor & model
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

        # 2️⃣ Switch to eval mode & half-precision
        self.model.eval()
        if use_fp16 and self.device == "cuda":
            self.model.half()  # FP16 weights

        # 3️⃣ (Optional) Compile for PyTorch 2.0+
        if use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.top_k = top_k

    def predict_caption(self, image: Image.Image) -> List[str]:
        image= image.resize((512,512))
        # 4️⃣ Preprocess
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 5️⃣ Inference under no_grad + autocast (if FP16)
        with torch.no_grad():
            if self.device == "cuda":
                # mixed-precision context
                autocast = torch.autocast if hasattr(torch, "autocast") else torch.cuda.amp.autocast
                with autocast(device_type="cuda"):
                    outputs = self.model.generate(
                        **inputs,
                        num_beams=7,  # 🔥
                        num_return_sequences=self.top_k, 
                        length_penalty=1.0,

                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    num_beams=self.top_k,
                    num_return_sequences=self.top_k,
                )

        # 6️⃣ Decode
        captions = [
            self.processor.decode(seq, skip_special_tokens=True)
            for seq in outputs
        ]
        return captions

