# smoke_test.py - quick synthetic data smoke test
import sys, os
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torch.nn as nn, torch.optim as optim
from types import SimpleNamespace
from torch.utils.data import DataLoader
from trainer import train, evaluate
from utils import set_seed

class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size=200, seq_len=16, embed_dim=16, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, input_ids=None, **kwargs):
        x = self.embed(input_ids)
        x = x.permute(0,2,1)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return SimpleNamespace(logits=logits)

def make_synthetic_loader(num_samples=40, seq_len=16, num_classes=2, batch_size=8):
    import torch
    X = torch.randint(0, 100, (num_samples, seq_len), dtype=torch.long)
    y = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
    examples = [{'input_ids': X[i], 'labels': y[i]} for i in range(num_samples)]
    def collate_fn(batch):
        return {'input_ids': torch.stack([b['input_ids'] for b in batch]), 'labels': torch.stack([b['labels'] for b in batch])}
    return DataLoader(examples, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = make_synthetic_loader()
    val_loader = make_synthetic_loader(num_samples=20)
    student = SimpleClassifier()
    teacher = SimpleClassifier(embed_dim=32)
    optimizer = optim.AdamW(student.parameters(), lr=1e-3)
    print('Starting smoke test on device:', device)
    history = train(student, teacher, train_loader, val_loader, optimizer, device=device, epochs=1)
    print('History:', history)
    metrics = evaluate(student, val_loader, device=device)
    print('Eval metrics:', metrics)

if __name__ == '__main__':
    main()
