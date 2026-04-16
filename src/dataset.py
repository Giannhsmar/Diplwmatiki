import torch
from torchvision import transforms, datasets


def load_data(data_path, batch_size):
    train_transforms = transforms.Compose([transforms.RandomRotation(5),
                                           transforms.Resize((256, 256)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])])
    train_data = datasets.ImageFolder(data_path, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return trainloader


if __name__ == '__main__':
    loader, data = load_data("data")
