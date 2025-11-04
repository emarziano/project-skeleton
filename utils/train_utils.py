import torch
from tqdm import tqdm
import wandb

def train_one_epoch(epoch, model, loader, criterion, optimizer, device="cuda"):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for _, (inputs, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    loss_avg = running_loss / len(loader)
    wandb.log({"train/loss": loss_avg, "train/accuracy": acc, "epoch": epoch})
    return loss_avg, acc


@torch.no_grad()
def validate(epoch, model, loader, criterion, device="cuda"):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    loss_avg = val_loss / len(loader)
    wandb.log({"val/loss": loss_avg, "val/accuracy": acc, "epoch": epoch})
    return loss_avg, acc
