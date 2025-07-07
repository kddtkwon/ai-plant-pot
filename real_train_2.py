import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.optim as optim

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# ✅ 경로
ROOT = r"/home/kwon/tomato/New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = os.path.join(ROOT, "real")   # 진짜 학습 데이터
VALID_DIR = os.path.join(ROOT, "valid")  # 실제 검증 데이터

print(f"✅ Train: {TRAIN_DIR}")
print(f"✅ Valid: {VALID_DIR}")

# ✅ 하이퍼파라미터
EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001
PATIENCE = 5

# ✅ Transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def main():
    # ✅ real → train, valid → 검증
    train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    valid_data = datasets.ImageFolder(VALID_DIR, transform=valid_transform)

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    class_names = train_data.classes
    print(f"✅ 클래스 목록 ({len(class_names)}개): {class_names}")

    base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = base_model.fc.in_features

    base_model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, len(class_names))
    )
    model = base_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    from torch.amp import GradScaler, autocast
    scaler = GradScaler(device='cuda')

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(valid_loader)
        val_acc = 100 * correct / total
        epoch_time = time.time() - start_time

        print(f"[Epoch {epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            MODEL_PATH = os.path.join(ROOT, "plant_resnet18_real_valid_best.pt")
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ 모델 저장 완료 (갱신): {MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"❗ EarlyStopping patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("⏹️ EarlyStopping 발동, 학습 종료")
                break

if __name__ == "__main__":
    main()

