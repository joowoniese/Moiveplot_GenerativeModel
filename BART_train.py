import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import random

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# JSON 데이터 로드 함수
def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

# Custom Dataset
class MovieDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        overview = item.get("overview", "")
        keywords = item.get("keywords", [])

        if not isinstance(keywords, list):
            keywords = []

        # 랜덤으로 키워드 3개 선택 (키워드가 3개 미만일 경우 모두 사용)
        selected_keywords = random.sample(keywords, min(3, len(keywords)))

        input_text = ", ".join(selected_keywords) if selected_keywords else "None"  # Input으로 키워드 3개 사용
        target_text = overview  # Output으로 overview 사용

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
        }

# 이후 코드는 그대로 사용



# JSON 데이터 로드 및 크기 확인
json_path = "/home/joowoniese/movieplot/moviedata/merged_movies_with_titles.json"
data = load_data(json_path)
print(f"Total samples in dataset: {len(data)}")

# Train-Test Split
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
print(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}")

# Tokenizer 및 모델 초기화
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Dataset 및 DataLoader 생성
train_dataset = MovieDataset(train_data, tokenizer)
val_dataset = MovieDataset(val_data, tokenizer)

batch = 16

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch)

# Training 함수
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:  # 매 10개 배치마다 로그 출력
            print(f"Batch {i + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


# Validation 함수
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)


# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(0)}")

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)

epochs = 100
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loss = train_model(model, train_loader, optimizer, device)
    val_loss = evaluate_model(model, val_loader, device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

# 모델 저장
model.save_pretrained(f"./BART_Keyword_Model/bart_3keyword_epoch{epochs}_batch{batch}_model")
tokenizer.save_pretrained(f"./BART_Keyword_Tokenizer/bart_3keyword_epoch{epochs}_batch{batch}_model")
print("Model saved successfully!")
