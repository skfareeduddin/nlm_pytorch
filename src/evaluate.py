import torch
import math

def calculate_perplexity(model, data_loader, vocab_size, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    perplexity = math.exp(avg_loss)
    return perplexity
