import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

from dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

MODEL_NAME = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

DATA_PATHS = {
    "FakeNews-Kaggle": "../data/processed/processed_FakeNews.csv",
    "LIAR": "../data/processed/processed_LIAR.csv",
    "ISOT": "../data/processed/processed_ISOT.csv"
}

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

train_dataloaders = {name: get_dataloader(file_path, tokenizer, batch_size=BATCH_SIZE, split="train") for name, file_path in DATA_PATHS.items()}
val_dataloaders = {name: get_dataloader(file_path, tokenizer, batch_size=BATCH_SIZE, split="val") for name, file_path in DATA_PATHS.items()}
test_dataloaders = {name: get_dataloader(file_path, tokenizer, batch_size=BATCH_SIZE, split="test") for name, file_path in DATA_PATHS.items()}

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

os.makedirs("../model", exist_ok=True)

def train_model(dataloader, dataset_name):
    model.train()
    print(f"Training on dataset: {dataset_name}")
    
    for epoch in range(EPOCHS):
        total_loss, correct, total = 0, 0, 0

        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False):
            optimizer.zero_grad()

            inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
        model_save_path = f"../model/{dataset_name}_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved: {model_save_path}")
    return accuracy

def validate_model(dataloader, dataset_name):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validating {dataset_name}", leave=False):
            inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Validation Accuracy on {dataset_name}: {accuracy:.2f}%")
    return accuracy

results = {}
for name, dataloader in train_dataloaders.items():
    train_acc = train_model(dataloader, name)
    val_acc = validate_model(val_dataloaders[name], name)
    test_acc = validate_model(test_dataloaders[name], name)
    
    results[name] = {
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc
    }
    
    print(f"Final Results for {name} -> Train: {train_acc:.2f}%, Validation: {val_acc:.2f}%, Test: {test_acc:.2f}%")

os.makedirs("../model", exist_ok=True)
with open("../model/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to ../model/results.json")

print("Model training complete & results saved!")