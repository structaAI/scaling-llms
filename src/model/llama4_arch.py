import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Dict, List, Tuple, Union, cast

from src.config.model_config import ModelConfig
from src.model.embedding import LLaMaEmbeddings
from src.model.transformer_block import LLaMaDecoder
from src.utils.rms_norm import RMSNorm

class CodeLLaMa(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.config = config
    self.padding_idx = config.pad_token_id

    self.embed_tokens = LLaMaEmbeddings(config)
    self.layers = nn.ModuleList([
        LLaMaDecoder(config) for _ in range(config.num_hidden_layers)
    ])
    self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    self.gradient_checkpointing = False
    self.post_init()
  
  def post_init(self):
    self.apply(self._init_weights)
  
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
      if module.bias is not None:
        torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
      if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()
  
  def forward(
      self, 
      input_ids: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.Tensor] = None,
      past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,  # Fixed: Allow None in list
      output_attentions: bool = False,
      output_hidden_states: bool = False
  ) -> Dict[str, Any]:
    batch_size, seq_len = input_ids.shape

    if attention_mask is None:
      attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
    
    if position_ids is None:
      position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
      position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    hidden_states = self.embed_tokens(input_ids=input_ids)
    
    all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = () if output_hidden_states else None
    all_attentions: Optional[Tuple[Optional[torch.Tensor], ...]] = () if output_attentions else None
    next_decoder_cache: Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], ...]] = () if self.config.use_cache else None
    
    if past_key_values is None:
      past_key_values = cast(List[Optional[Tuple[torch.Tensor, torch.Tensor]]], [None] * len(self.layers))
    
    for idx, decoder_layer in enumerate(self.layers):
      if output_hidden_states and all_hidden_states is not None:
        all_hidden_states = all_hidden_states + (hidden_states,)
        
      layer_outputs = decoder_layer(
          hidden_states=hidden_states,
          attention_mask=attention_mask,
          position_ids=position_ids,
          past_key_value=past_key_values[idx],
          output_attentions=output_attentions
      )
        
      hidden_states = layer_outputs[0]
        
      if output_attentions and all_attentions is not None:
          # Handle case where attention weights might not be returned
        if len(layer_outputs) > 1 and layer_outputs[1] is not None:
            all_attentions = all_attentions + (layer_outputs[1],)
        else:
            all_attentions = all_attentions + (None,)
      
      if self.config.use_cache and next_decoder_cache is not None:
        # Handle case where past_key_value might not be returned
        if len(layer_outputs) > 2 and layer_outputs[2] is not None:
            next_decoder_cache = next_decoder_cache + (layer_outputs[2],)
        else:
            next_decoder_cache = next_decoder_cache + (None,)
    
    # Apply final normalization
    hidden_states = self.norm(hidden_states)
    
    if output_hidden_states and all_hidden_states is not None:
      all_hidden_states = all_hidden_states + (hidden_states,)
    
    # Fixed: Return proper dictionary with all outputs
    return {
      "last_hidden_state": hidden_states,
      "hidden_states": all_hidden_states,
      "attentions": all_attentions,
      "past_key_values": next_decoder_cache
    }


class LLaMaForCausalLM(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.config = config
    self.model = CodeLLaMa(config)
    self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    if self.config.tie_word_embeddings:
        self.lm_head.weight = self.model.embed_tokens.word_embeddings.weight
    
    self.post_init()

  def post_init(self):
    self.apply(self._init_weights)
  
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
      if module.bias is not None:
          torch.nn.init.constant_(module.bias, 0)
  
  def forward(
      self,
      input_ids: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.Tensor] = None,
      past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,  # Fixed: Allow None in list
      labels: Optional[torch.Tensor] = None,
      **kwargs
  ) -> Dict[str, Any]:
      # Pass output_attentions and output_hidden_states from kwargs
    output_attentions = kwargs.get('output_attentions', False)
    output_hidden_states = kwargs.get('output_hidden_states', False)
    
    outputs = self.model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_values=past_key_values,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states
    )

    hidden_states = outputs["last_hidden_state"]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
      shift_logits = logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return {
      "loss": loss,
      "logits": logits,
      "hidden_states": outputs.get("hidden_states"),
      "attentions": outputs.get("attentions"),
      "past_key_values": outputs.get("past_key_values")
    }