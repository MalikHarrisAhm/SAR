import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional
from collections import defaultdict
import numpy as np


@dataclass
class TokenWeightConfig:
    hidden_dim: int = 256
    memory_size: int = 10000
    min_weight: float = 0.1
    max_weight: float = 5.0
    importance_lr: float = 0.01
    token_weights: Dict[Union[str, int], float] = field(default_factory=dict)
    default_weight: float = 1.0
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    update_interval: int = 50


class ImportanceMemory:
    def __init__(self, config: TokenWeightConfig):
        self.config = config
        self.short_term = defaultdict(list)
        self.long_term = defaultdict(float)
        self.token_performance = defaultdict(list)
        self.update_counter = 0
        self.token_gradients = defaultdict(list)

    def update(self, token_ids: torch.Tensor, loss_improvement: float,
               gradient_info: torch.Tensor, step: int):
        self.update_counter += 1

        # Flatten token_ids if it's 2D
        token_ids = token_ids.view(-1)
        unique_tokens = token_ids.unique()

        # Ensure gradient_info is properly shaped and normalized
        grad_norms = torch.norm(gradient_info, dim=-1)  # [batch_size, seq_len]
        grad_norms = grad_norms.view(-1)  # Flatten to [batch_size * seq_len]
        grad_norms = F.normalize(grad_norms, dim=-1)

        for token_id in unique_tokens:
            token_id = token_id.item()
            # Create mask matching the flattened dimensions
            token_mask = (token_ids == token_id)
            token_grads = grad_norms[token_mask]

            # Safe averaging - handle empty tensors
            if token_grads.numel() > 0:
                avg_grad = token_grads.mean().item()
            else:
                avg_grad = 0.0

            importance = loss_improvement * avg_grad
            warmup_scale = min(1.0, step / self.config.warmup_steps)
            importance *= warmup_scale

            # Update short-term memory
            self.short_term[token_id].append(importance)
            if len(self.short_term[token_id]) > self.config.memory_size:
                self.short_term[token_id].pop(0)

            # Update long-term memory periodically
            if self.update_counter % self.config.update_interval == 0:
                if self.short_term[token_id]:  # Check if we have any values
                    current_importance = np.mean(self.short_term[token_id])
                    self.long_term[token_id] = (
                            (1 - self.config.importance_lr) * self.long_term[token_id] +
                            self.config.importance_lr * current_importance
                    )

            # Update gradient history
            self.token_gradients[token_id].append(avg_grad)
            if len(self.token_gradients[token_id]) > self.config.memory_size:
                self.token_gradients[token_id].pop(0)

    def get_importance(self, token_id: int) -> float:
        base_importance = self.long_term[token_id]
        if self.token_gradients[token_id]:
            recent_grads = self.token_gradients[token_id][-100:]
            grad_volatility = np.std(recent_grads) if len(recent_grads) > 1 else 0
            base_importance *= (1 + grad_volatility)
        return float(base_importance)


class EnhancedMetaWeightNetwork(nn.Module):
    def __init__(self, hidden_size: int, config: TokenWeightConfig):
        super().__init__()
        self.config = config
        self.importance_memory = ImportanceMemory(config)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.meta_network = nn.Sequential(
            nn.Linear(hidden_size * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        # Change to interpolation-based positional embeddings
        self.max_position_embeddings = 2048
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.max_position_embeddings, hidden_size)
        )

        self.manual_weights = {}
        if config.token_weights:
            self.set_token_weights(config.token_weights)

    def get_position_embeddings(self, seq_length):
        if seq_length <= self.max_position_embeddings:
            return self.position_embedding[:, :seq_length, :]
        else:
            # Use interpolation for sequences longer than max_position_embeddings
            pos_embed = self.position_embedding.squeeze(0)
            pos_embed = nn.functional.interpolate(
                pos_embed.transpose(0, 1).unsqueeze(0),
                size=seq_length,
                mode='linear',
                align_corners=False
            )
            return pos_embed.squeeze(0).transpose(0, 1).unsqueeze(0)

    def set_token_weights(self, token_weights: Dict[Union[str, int], float]):
        self.manual_weights = token_weights

    def get_token_weight(self, token_id: int, predicted_weight: float) -> float:
        if token_id in self.manual_weights:
            return self.manual_weights[token_id]
        return torch.clamp(
            predicted_weight,
            self.config.min_weight,
            self.config.max_weight
        )

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, hidden_size = hidden_states.shape

        # Get interpolated position embeddings for the current sequence length
        pos_embeddings = self.get_position_embeddings(seq_length)

        # Add positional embeddings
        hidden_states = hidden_states + pos_embeddings.to(hidden_states.device)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length),
                device=hidden_states.device,
                dtype=torch.bool
            )
        else:
            # Convert attention mask to boolean type
            attention_mask = attention_mask.to(torch.bool)

        # Handle attention mechanism
        attended_states, _ = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=~attention_mask,  # Invert the mask for key_padding_mask
            need_weights=False
        )

        combined_states = torch.cat([hidden_states, attended_states], dim=-1)
        base_weights = self.meta_network(combined_states).squeeze(-1)
        final_weights = torch.zeros_like(base_weights)

        for b in range(batch_size):
            for s in range(seq_length):
                if attention_mask[b, s]:
                    token_id = token_ids[b, s].item()
                    predicted = base_weights[b, s]
                    importance = self.importance_memory.get_importance(token_id)
                    weight = predicted * (1 + importance)
                    final_weights[b, s] = self.get_token_weight(token_id, weight)

        return final_weights


class TokenWeightedTrainer(Trainer):
    def __init__(
            self,
            model: nn.Module,
            args: TrainingArguments,
            meta_network=None,
            **kwargs
    ):
        self.meta_network = meta_network
        if 'meta_network' in kwargs:
            del kwargs['meta_network']
        super().__init__(model=model, args=args, **kwargs)
        self.prev_loss = None
        self.step_counter = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        attention_mask = inputs.get("attention_mask")
        token_weights = self.meta_network(
            hidden_states,
            inputs["input_ids"],
            attention_mask
        )

        labels = inputs.get("labels")
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = token_weights[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        loss = loss.view(shift_labels.size())
        weighted_loss = (loss * shift_weights).mean()

        if self.prev_loss is not None:
            loss_improvement = self.prev_loss - weighted_loss.item()
            self.meta_network.importance_memory.update(
                inputs["input_ids"].view(-1),
                loss_improvement,
                hidden_states.detach(),
                self.step_counter
            )

        self.prev_loss = weighted_loss.item()
        self.step_counter += 1

        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def log(self, logs: Dict[str, float], start_time=None, **kwargs) -> None:
        """Ensure we log evaluation metrics"""
        if "eval_loss" in logs:
            logs = {**logs, "step": self.state.global_step}
        super().log(logs, start_time=start_time, **kwargs)  # Changed from start_time to start_time=start_time

    def evaluation_loop(
            self,
            dataloader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ):
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix
        )

        if self.is_world_process_zero():
            print(f"\nEvaluation metrics at step {self.state.global_step}:")
            for key, value in eval_output.metrics.items():
                print(f"{key}: {value:.4f}")

        return eval_output