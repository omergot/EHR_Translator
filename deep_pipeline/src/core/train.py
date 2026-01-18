import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..adapters.yaib import YAIBRuntime
from ..core.translator import Translator


class TranslatorTrainer:
    def __init__(
        self,
        yaib_runtime: YAIBRuntime,
        translator: Translator,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.yaib_runtime = yaib_runtime
        self.translator = translator.to(device)
        self.device = device
        self.optimizer = Adam(self.translator.parameters(), lr=learning_rate)
        
        self.best_val_loss = float('inf')
        self.best_translator_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.translator.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = tuple(b.to(self.device) for b in batch)
            
            self.optimizer.zero_grad()
            
            translated_data = self.translator(batch)
            baseline_outputs = self.yaib_runtime.forward((translated_data, batch[1], batch[2]))
            loss = self.yaib_runtime.compute_loss(baseline_outputs, (translated_data, batch[1], batch[2]))
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.translator.eval()
        all_outputs = []
        all_batches = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(b.to(self.device) for b in batch)
                translated_data = self.translator(batch)
                baseline_outputs = self.yaib_runtime.forward((translated_data, batch[1], batch[2]))
                all_outputs.append(baseline_outputs)
                all_batches.append(batch)
        
        metrics = {}
        for outputs, batch in zip(all_outputs, all_batches):
            batch_metrics = self.yaib_runtime.compute_metrics(outputs, batch, split="val")
            for key, value in batch_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
        return avg_metrics
    
    def train(
        self,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Optional[Path] = None,
        patience: int = 10,
    ):
        logging.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            logging.info(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUCROC: {val_metrics['AUCROC']:.4f}, "
                f"Val AUCPR: {val_metrics['AUCPR']:.4f}"
            )
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_translator_state = self.translator.state_dict().copy()
                logging.info(f"New best validation loss: {self.best_val_loss:.4f}")
                
                if checkpoint_dir:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / "best_translator.pt"
                    torch.save({
                        'epoch': epoch,
                        'translator_state_dict': self.best_translator_state,
                        'val_loss': self.best_val_loss,
                        'val_metrics': val_metrics,
                    }, checkpoint_path)
                    logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        if self.best_translator_state:
            self.translator.load_state_dict(self.best_translator_state)
            logging.info("Loaded best translator weights")



