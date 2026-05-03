import torch.nn as nn
from transformers import Trainer

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.BCEWithLogitsLoss()

        loss = loss_fn(logits, labels)

        if return_outputs:
            return loss, outputs
        return loss