import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn

# ==========================================
# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# ==========================================
# ✅ 클래스명 (학습셋과 100% 일치해야 함)
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Mosaic_virus",
    "Tomato___Healthy"
]

# ==========================================
# ✅ 저장된 학습 모델 경로
MODEL_PATH = "/home/kwon/tomato/New Plant Diseases Dataset(Augmented)/plant_resnet18_colab_best.pt"


# ==========================================
# ✅ 이미지 전처리 transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================================
# ✅ 모델 복원
base_model = models.resnet18(weights=None)
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, len(class_names))
)
base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = base_model.to(device)
model.eval()

print(f"✅ 모델 로드 완료: {MODEL_PATH}")

# ==========================================
# ✅ OpenCV 웹캠 켜기
cap = cv2.VideoCapture(0)  # 내장 웹캠 (필요하면 외장캠 번호 1, 2로 바꿔도 OK)

if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

print("✅ 웹캠 시작! [Q] 누르면 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 캡처 실패")
        break

    # BGR → RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        top3_prob, top3_idx = torch.topk(probs, 3)

    # ✅ Top-3 예측 출력
    for i in range(3):
        label = class_names[top3_idx[0][i]]
        prob = top3_prob[0][i].item() * 100
        text = f"{i+1}: {label} ({prob:.1f}%)"
        cv2.putText(frame,
                    text,
                    (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

    cv2.imshow('Tomato Leaf Top-3 Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

