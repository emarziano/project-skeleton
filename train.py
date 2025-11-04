from data.prepare_tinyimagenet import prepare_tinyimagenet
from data.dataloader_tinyimagenet import get_tinyimagenet_dataloaders
from models.custom_net import CustomNet
from utils.train_utils import train_one_epoch, validate
from utils.wandb_utils import setup_wandb
import torch
import os
from torch import nn, optim

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "momentum": 0.9,
        "model": "CustomNet",
    }

    setup_wandb("tiny-imagenet-training", config)

    prepare_tinyimagenet()  # scarica e prepara dataset
    train_loader, val_loader = get_tinyimagenet_dataloaders(batch_size=config["batch_size"])

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    best_acc = 0

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch: {epoch}")
        train_one_epoch(epoch, model, train_loader, criterion, optimizer)

        # Validation step
        val_accuracy = validate(model, val_loader, criterion)

        # 🔽 Aggiungi QUI il salvataggio del modello migliore
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"💾 Saved best model with acc={best_acc:.2f}%")

    print(f"✅ Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
