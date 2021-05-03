from PIL import Image
from torchvision import transforms


transformations = transforms.Compose([
            # Required - https://pytorch.org/hub/pytorch_vision_inception_v3/
            transforms.Resize((299, 229)),
            transforms.ToTensor(),
            # Required - https://pytorch.org/hub/pytorch_vision_inception_v3/
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def transform(image_path):
    return transformations(Image.open(image_path).convert("RGB"))
