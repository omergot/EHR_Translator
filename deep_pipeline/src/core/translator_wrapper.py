import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from typing import Tuple

from ..adapters.yaib import YAIBRuntime
from ..core.translator import Translator


class TranslatorModelWrapper(LightningModule):
    def __init__(
        self,
        translator: Translator,
        yaib_runtime: YAIBRuntime,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.translator = translator
        self.yaib_runtime = yaib_runtime
        self.learning_rate = learning_rate
        self.requires_backprop = True
        
        self.yaib_runtime.load_baseline_model()
        for param in self.yaib_runtime._model.parameters():
            param.requires_grad = False
        self.yaib_runtime._model.eval()
    
    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        translated_data = self.translator(batch)
        baseline_outputs = self.yaib_runtime.forward((translated_data, batch[1], batch[2]))
        return baseline_outputs
    
    def step_fn(self, element, step_prefix=""):
        data, labels, mask = element[0], element[1], element[2]
        batch = (data, labels, mask)
        outputs = self(batch)
        loss = self.yaib_runtime.compute_loss(outputs, batch)
        
        if hasattr(self.yaib_runtime._model, 'metrics') and step_prefix in self.yaib_runtime._model.metrics:
            prediction = torch.masked_select(outputs, mask.unsqueeze(-1)).reshape(-1, outputs.shape[-1])
            target = torch.masked_select(labels, mask)
            
            for key, value in self.yaib_runtime._model.metrics[step_prefix].items():
                if hasattr(value, 'update'):
                    if outputs.shape[-1] > 1:
                        prediction_proba = torch.softmax(prediction, dim=-1)[:, 1]
                    else:
                        prediction_proba = torch.sigmoid(prediction)
                    value.update(prediction_proba, target)
        
        self.log(f"{step_prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step_fn(batch, step_prefix="train")
    
    def validation_step(self, batch, batch_idx):
        return self.step_fn(batch, step_prefix="val")
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.translator.parameters(), lr=self.learning_rate)


