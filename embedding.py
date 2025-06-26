import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

def EnzEmb(sequence = None, model_name = "facebook/esm2_t6_8M_UR50D"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling over the sequence
    return embeddings