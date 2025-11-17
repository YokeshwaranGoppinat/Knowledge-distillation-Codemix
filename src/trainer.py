# trainer.py (minimal safe version)
import torch, torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    ce = F.cross_entropy(student_logits, labels)
    t = temperature
    p_student = F.log_softmax(student_logits / t, dim=-1)
    p_teacher = F.softmax(teacher_logits / t, dim=-1)
    kld = F.kl_div(p_student, p_teacher, reduction='batchmean') * (t*t)
    return alpha * ce + (1-alpha) * kld

def train(model, teacher_model, train_loader, val_loader, optimizer, scheduler=None, device='cpu', epochs=1):
    device = torch.device('cuda' if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
    model.to(device)
    if teacher_model is not None:
        teacher_model.to(device)
        teacher_model.eval()
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[], 'val_f1':[]}
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = to_device(batch, device)
            labels = batch.pop('labels')
            optimizer.zero_grad()
            outputs = model(**batch)
            s_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            if teacher_model is not None:
                with torch.no_grad():
                    t_out = teacher_model(**batch)
                    t_logits = t_out.logits if hasattr(t_out, 'logits') else t_out[0]
                loss = distillation_loss(s_logits, t_logits, labels)
            else:
                loss = F.cross_entropy(s_logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        history['train_loss'].append(running / max(1, len(train_loader)))
        # simple eval
        model.eval()
        preds, trues = [], []
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, device)
                labels = batch.pop('labels')
                out = model(**batch)
                logits = out.logits if hasattr(out, 'logits') else out[0]
                val_losses.append(F.cross_entropy(logits, labels).item())
                preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                trues.extend(labels.cpu().numpy().tolist())
        val_loss = sum(val_losses)/max(1, len(val_losses))
        val_acc = accuracy_score(trues, preds) if trues else 0.0
        val_f1 = f1_score(trues, preds, average='weighted') if trues else 0.0
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
    return history

def evaluate(model, loader, device='cpu'):
    device = torch.device('cuda' if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
    model.to(device).eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop('labels')
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(**batch)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    from sklearn.metrics import accuracy_score, f1_score
    return {'accuracy': accuracy_score(trues,preds) if trues else 0.0, 'f1': f1_score(trues,preds, average='weighted') if trues else 0.0}
