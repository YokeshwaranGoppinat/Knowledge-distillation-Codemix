# model_utils.py (minimal safe version)
from transformers import AutoModelForSequenceClassification, AutoConfig

def build_model(model_name='distilbert-base-uncased', num_labels=2, freeze_base=False):
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    if freeze_base:
        for n,p in model.named_parameters():
            if 'classifier' not in n and 'pooler' not in n:
                p.requires_grad = False
    return model
