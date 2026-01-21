import torch
import torch.nn as nn
from typing import Tuple


class Translator(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
    
    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError


class IdentityTranslator(Translator):
    def __init__(self, input_size: int):
        super().__init__(input_size)
        self.input_size = input_size
        # Dummy learnable scalar so optimizer isn't empty
        self._dummy = nn.Parameter(torch.zeros(()))
    
    def forward(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        data = batch[0]
        return data + (self._dummy * 0.0)





