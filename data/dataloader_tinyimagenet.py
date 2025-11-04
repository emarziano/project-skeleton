import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_tinyimagenet_dataloaders(root="dataset/tiny-imagenet-200", batch_size=32, num_workers=2):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolder(root=f"{root}/train", transform=transform)
    val_ds = ImageFolder(root=f"{root}/val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
