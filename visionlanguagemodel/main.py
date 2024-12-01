import torch
from PIL import Image
from models.encoder import EfficientNetB4Encoder
from models.text_generator import GPT2TextGenerator
from torchvision import transforms

# Initialize models
image_encoder = EfficientNetB4Encoder()
text_generator_stage1 = GPT2TextGenerator("gpt2")
text_generator_stage2 = GPT2TextGenerator("gpt2")

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_radiology_report(image_path):
    # Step 1: Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)

    # Step 2: Image encoding
    with torch.no_grad():
        encoded_features = image_encoder(image_tensor)

    # Step 3: Stage 1 text generation
    initial_prompt = "The radiology image analysis suggests"
    stage1_output = text_generator_stage1.generate_text(initial_prompt + str(encoded_features.tolist()), max_length=50)

    # Step 4: Stage 2 text generation
    updated_prompt = f"{stage1_output}. Further details include"
    final_report = text_generator_stage2.generate_text(updated_prompt, max_length=100)

    return final_report

if __name__ == "__main__":
    image_path = "example_radiology_image.jpg"  # Replace with your image path
    report = generate_radiology_report(image_path)
    print("Generated Radiology Report:")
    print(report)
