import os
import io
import csv
import torch
from torchvision import models, transforms
from PIL import Image

# ================== PATH FIX (MOST IMPORTANT) ==================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'multi_class_resnet18_diagnosis_v2.pth')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'model_class_names.txt')
LOOKUP_CSV_PATH = os.path.join(BASE_DIR, 'recommendation_lookup.csv')

# ===============================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global variables
model = None
lookup_table = {}
class_names = []

# ------------------ FILE CHECK ------------------

def check_required_files():
    required = [MODEL_PATH, CLASS_NAMES_PATH, LOOKUP_CSV_PATH]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing required files:")
        for f in missing:
            print("   -", f)
        return False
    return True

# ------------------ LOAD MODEL + LOOKUP ------------------

def load_data():
    global model, lookup_table, class_names

    if not check_required_files():
        return False

    print(f"✅ Loading model from: {MODEL_PATH}")
    print(f"📦 Using device: {DEVICE}")

    # Load class names
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]

    num_classes = len(class_names)

    # Load model
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"✅ Model loaded successfully ({num_classes} classes)")

    # Load recommendations CSV
    with open(LOOKUP_CSV_PATH, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        clean_to_data = {row['Clean_Name']: row for row in reader}

    for raw_name in class_names:
        clean_name = raw_name.replace('___', ' - ').replace('_', ' ').title().strip()

        if clean_name == 'Corn - Cercospora Leaf Spot Gray Leaf Spot':
            clean_name = 'Corn - Cercospora/Gray Leaf Spot'
        if clean_name == 'Orange - Haunglongbing (Citrus Greening)':
            clean_name = 'Orange - Citrus Greening (HLB)'

        lookup_table[raw_name] = clean_to_data.get(clean_name, {
            'Clean_Name': clean_name,
            'Recommendation': 'No specific recommendation found.'
        })

    print(f"✅ Recommendation table loaded ({len(lookup_table)} entries)")
    return True

# ------------------ IMAGE PREPROCESS ------------------

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = data_transforms(image).unsqueeze(0)
    return tensor

# ------------------ PREDICTION ------------------

def predict_diagnosis(image_tensor):
    if model is None:
        raise RuntimeError("Model is not loaded")

    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, idx = torch.max(probs, 0)

        raw_name = class_names[idx.item()]
        data = lookup_table.get(raw_name)

        diagnosis = data['Clean_Name']
        recommendation = data['Recommendation']
        confidence_pct = f"{confidence.item() * 100:.2f}%"

        return diagnosis, confidence_pct, recommendation
