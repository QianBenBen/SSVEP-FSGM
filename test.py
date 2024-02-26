import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import Benchmark
from model import EEGNet, DepthwiseSeparableConv2d

def eeg_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = Benchmark("E:\Datasets\BCI\SSVEP\Benchmark", train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                          ]))

    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
    # 必须要可以访问倒EEGNet结构
    eegnet = torch.load("Weights/eeg_gpu.pth")
    eegnet.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = eegnet(inputs)
            values, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += len(labels)

        accuracy = total_correct / total_samples
        print(f"Test Accuracy: {accuracy:.4f}")