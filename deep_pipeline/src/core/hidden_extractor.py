"""Extract penultimate-layer hidden states from frozen baseline via forward hooks."""
import logging

import torch
import torch.nn as nn


class HiddenStateExtractor:
    """Registers a forward hook on the penultimate layer of a baseline model
    to capture hidden representations during forward passes.

    Usage:
        extractor = HiddenStateExtractor(model)
        _ = model(input)  # triggers hook
        hidden = extractor.hidden_states  # (B, T, hidden_dim)
        extractor.remove()  # cleanup
    """

    def __init__(self, model: nn.Module):
        self.hidden_states: torch.Tensor | None = None
        self._hook = self._find_and_hook(model)

    def _find_and_hook(self, model: nn.Module):
        # YAIB LSTM: model.rnn (nn.LSTM) -> model.logit (nn.Linear)
        if hasattr(model, "rnn"):
            target = model.rnn
        elif hasattr(model, "lstm"):
            target = model.lstm
        else:
            children = list(model.children())
            target = children[-2] if len(children) > 1 else children[0]
        logging.info("[hidden-extractor] Hooked layer: %s", type(target).__name__)
        return target.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # LSTM output: (seq_output, (h_n, c_n))
        if isinstance(output, tuple):
            self.hidden_states = output[0]  # (B, T, hidden_dim)
        else:
            self.hidden_states = output

    def remove(self):
        if self._hook:
            self._hook.remove()
            self._hook = None
