import torch
import torch.nn as nn
from transformers import RobertaModel

class FakeNewsClassifier(nn.Module):
    def __init__(self, model_name="roberta-base", num_classes=2):
        super(FakeNewsClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped_out = self.dropout(pooled_output)
        return self.classifier(dropped_out)

# Function to load model with pre-trained weights
def load_model(model_path, model_name="roberta-base", num_classes=2, device="cpu"):
    model = FakeNewsClassifier(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
