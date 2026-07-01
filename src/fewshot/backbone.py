import os
import logging
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from src.fewshot.prompt import (
    get_prompt_lettre, 
    decode_prompt_lettre, 
    get_prompt_class,
)

logger = logging.getLogger(__name__)

# You can remoove, avoid some pytorch import errors
try:
    import torchao.quantization as tq
    class Float8WeightOnlyConfig: pass
    class Float8DynamicActivationFloat8WeightConfig: pass
    tq.Float8WeightOnlyConfig = Float8WeightOnlyConfig
    tq.Float8DynamicActivationFloat8WeightConfig = Float8DynamicActivationFloat8WeightConfig
except ImportError:
    pass

#import open_clip
from transformers import CLIPTokenizerFast
from transformers import AutoProcessor, AutoModel
from transformers import Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration

# --------------
#      CLIP  
# --------------

class ClipWrapper(nn.Module):
    def __init__(
            self, 
            model_name, 
            weights,
            num_features_hidden,
            num_features_out,
            **kwargs
        ):
        super().__init__()
        self.type_model = 'CLIP'

        if model_name == 'DFN':
            #from transformers import CLIPTokenizerFast
            import open_clip

            ckpt = os.path.join(weights, "open_clip_pytorch_model.bin")
            base, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained=ckpt, device="cpu")
            base.eval()
            base.visual.output_tokens = True

            class _DFNWrap(torch.nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.base = base
                    self.visual = base.visual
                def encode_image(self, x):
                    pooled, tokens = self.visual(x)          # pooled already projected
                    tokens = self.visual.ln_post(tokens)
                    tokens = tokens @ self.visual.proj
                    return pooled, tokens
                def encode_text(self, x):
                    return self.base.encode_text(x)

            self.model = _DFNWrap(base)
            self.model.eval()

            self.normalization = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

            tok = CLIPTokenizerFast.from_pretrained(weights, local_files_only=True)
            self.tokenize = lambda texts, truncate=False, context_length=77: tok(
                texts, padding="max_length", max_length=context_length, truncation=truncate, return_tensors="pt"
            )["input_ids"]
            self.context_length = 77

            self.depth = len(self.model.visual.transformer.resblocks)
            self.num_features_out = num_features_out
            self.num_features_hidden = num_features_hidden
        
        elif model_name == 'SigLIPv1':
            # https://chatgpt.com/c/6995c1bc-2b84-8393-a00c-0dd15f68e1f1
            #from transformers import AutoProcessor, AutoModel

            base = AutoModel.from_pretrained(weights, local_files_only=True)
            base.eval()

            class _SigLIPWrap(torch.nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.base = base
                    self.visual = base.vision_model

                def encode_image(self, x):
                    #logger.warning(f"Img Befor shape: {x.shape} {x.mean()} {x.std()}")
                    img = self.base.get_image_features(pixel_values=x, interpolate_pos_encoding=False)  # [B,768]
                    
                    #logger.warning(f"Img After shape: {img.shape} {img.mean()} {img.std()}")
                    return img, None

                def encode_text(self, input_ids, attention_mask=None):
                    #logger.warning(f"Txt Before shape: {input_ids.shape} {input_ids[0]}")
                    txt = self.base.get_text_features(input_ids=input_ids)  # [B,768]
                    #logger.warning(f"Txt After shape: {txt.shape} {txt.mean()} {txt.std()}")
                    return txt 

            self.model = _SigLIPWrap(base)
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
            self.normalization = Normalize(
                tuple(self.processor.image_processor.image_mean),
                tuple(self.processor.image_processor.image_std),
            )
            #self.normalization = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

            self.context_length = base.config.text_config.max_position_embeddings     # 64
            tok = self.processor.tokenizer
            self.tokenize = lambda texts, truncate=False, context_length=self.context_length: tok(
                texts, padding="max_length", max_length=context_length,
                truncation=truncate, return_tensors="pt", return_attention_mask=True
            )

            self.depth = len(self.model.visual.encoder.layers)
            self.num_features_out = num_features_out
            self.num_features_hidden = num_features_hidden
            self.need_training = None
       
        self.model_name = model_name

    def prep(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] float in [0,1] or [0,255] already cast by caller
        assert x.shape[-1] == 224 and x.shape[-2] == 224, "Input image size must be 224x224"
        x = self.normalization(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        # x: [B,3,H,W] float
        x = self.prep(x)

        cls_token, patch_tokens = self.model.encode_image(x)

        cls_token = F.normalize(cls_token, dim=-1, p=2)
        if patch_tokens is not None:
            patch_tokens = F.normalize(patch_tokens, dim=-1, p=2)

        x_dict = {
            "x_norm_clstoken": cls_token,    # [B, C]
            "x_storage_tokens": None,
            "x_norm_patchtokens": patch_tokens, # [B, T, C]
            "x_prenorm": None,
        }
        return x_dict

    def forward_text(self, x_text: torch.Tensor, **kwargs):

        device = next(self.model.parameters()).device
        # x: [B, seq_len] int64 tokens
        if self.model_name == "SigLIPv1":
            #logger.warning(x_text)
            txt = self.tokenize(x_text, truncate=False, context_length=self.context_length)
            txt_token = self.model.encode_text(
                input_ids=txt["input_ids"].to(device),
                attention_mask=txt["attention_mask"].to(device),
            )
            #logger.warning(f"Attetnion Mask: {txt['attention_mask'].shape} {txt['attention_mask'][0]}")
            #logger.warning(f"Txt Token: {txt_token.shape}")
        else:
            txt_token = self.tokenize(x_text, truncate=False, context_length=self.context_length).to(device)
            txt_token = self.model.encode_text(txt_token)

        txt_token = F.normalize(txt_token, dim=-1, p=2)

        x_dict = {
            "x_norm_txttoken": txt_token,    # [B, C]
        }
        return x_dict


# --------------
#      LMMs  
# --------------

class VLVMWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def init_task_specific_data(
            self, 
            support_set, 
            support_label, 
            label_to_str, 
            dataset_name, 
            prompt_cfg,
            benchmark_name,
        ):
        self.current_label_to_str = label_to_str
        self.current_dataset_name = dataset_name
        
        
        if prompt_cfg['type'] == "simple":
            self.current_text_prompt = prompt_cfg["prompt"]
            self.decode_output = decode_prompt_lettre

        elif prompt_cfg['type'] == "lettre":
            self.current_text_prompt, self.current_class_to_lettre = get_prompt_lettre(
                label_to_str, dataset_name, benchmark_name,
                skip_classes = prompt_cfg["skip_classes"],
            )
            self.decode_output = decode_prompt_lettre
            self.get_class_txt = self.forward_text_heads
        else:
            raise NotImplementedError(f"Unknown prompt template {prompt_cfg['type']}")
    
        self.current_text_prompt_class = get_prompt_class(
            label_to_str = label_to_str, 
            dataset_name = dataset_name, 
            benchmark_name = benchmark_name,
            current_text_prompt = self.current_text_prompt,
        )
        #logger.info(f"Prompt Image Mapping: {self.current_text_prompt}")
        #logger.info(f"Prompt Class Mapping: {self.current_text_prompt_class}")
    
    def prep(self, x: torch.Tensor) -> torch.Tensor:
        pass


class QwenV2Wrapper(VLVMWrapper):
    def __init__(
            self, 
            model_name, 
            weights,
            num_features_out,
            output_llm_lasttok_heads=False,
            **kwargs
        ):
        super().__init__()
        self.type_model = 'LMM'
        self.model_name = model_name
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            weights,
            local_files_only=True,
            attn_implementation="sdpa",
            dtype=torch.float16, 
            device_map=kwargs["device"],
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            weights
        )

        self.depth = len(self.model.model.visual.blocks)
        self.num_features_out = num_features_out
        self.num_features_hidden = 1280

        self.depth_llm = len(self.model.model.language_model.layers)
        self.num_features_hidden_llm = 3584

        def register_llm_attn_heads_last_token_hooks(model):
            acts = {}
            hooks = []
            layers = model.model.language_model.layers
        
            for i, layer in enumerate(layers):
                attn = layer.self_attn
                H = attn.num_heads
                Dh = attn.head_dim
        
                def _pre_o_proj_hook(module, inp, i=i, H=H, Dh=Dh):
                    y = inp[0]                      # [B, T, H*Dh] concat heads (post-attn, pre-o_proj)
                    acts[i] = y[:, -1, :].view(y.size(0), H, Dh).detach()  # [B, H, Dh]
        
                hooks.append(attn.o_proj.register_forward_pre_hook(_pre_o_proj_hook))
        
            def remove():
                for h in hooks:
                    h.remove()
        
            return acts, remove

        self.output_llm_lasttok_heads = output_llm_lasttok_heads

        self.register_llm_attn_heads_last_token_hooks   = register_llm_attn_heads_last_token_hooks
        

        def register_llm_last_layer_prompt_lasttok_hook(model):
            acts = {}
            layer = model.model.language_model.layers[-1]

            def f(module, inp, out):
                #hs = out[0]  # [B, T, H]  # WARNING: TRANSFORMER NOT RETROCOMPATIBLE
                hs = out  # [B, T, H]  # WARNING: TRANSFORMER NOT RETROCOMPATIBLE
                if hs.shape[1] > 1 and "x_norm_clstoken" not in acts:  # first (prompt) pass in generate
                    acts["x_norm_clstoken"] = hs[:, -1, :].detach()     # [B, H]

            h = layer.register_forward_hook(f)

            def remove():
                h.remove()

            return acts, remove

        self.register_llm_last_layer_prompt_lasttok_hook = register_llm_last_layer_prompt_lasttok_hook

        self.need_training = None

    def forward(self, x: torch.Tensor, text_prompt=None):

        B = x.shape[0]
        device = x.device

        # Build Text Prompt
        # 1 format classes
        if text_prompt is None:
            text_prompt = self.current_text_prompt

        #logger.info(f"Prompt: {text_prompt}")

        # 2 format chat
        text_prompt = self.processor.apply_chat_template(
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "Any"},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ], 
            tokenize=False, 
            add_generation_prompt=True,
        )

        # Build Input
        inputs = self.processor(text=[text_prompt]*B, images=255*x, padding=True, return_tensors="pt")
        inputs.to(device)


        # Get Output Hooks
        if self.output_llm_lasttok_heads:
            acts_llm_last_heads, remove_llm_last_heads = self.register_llm_attn_heads_last_token_hooks(self.model)
        
        acts_cls, remove_cls = self.register_llm_last_layer_prompt_lasttok_hook(self.model)

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=1)

        # Remove Hooks
        output_dictionary = {}
        if self.output_llm_lasttok_heads:
            output_dictionary["attn_heads"] = torch.stack([acts_llm_last_heads[i] for i in range(self.depth_llm)], dim=1)   
            remove_llm_last_heads()
        output_dictionary["x_norm_clstoken"] = acts_cls["x_norm_clstoken"]  # [B, 3584]
        remove_cls()

        
        # Post-Process Output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        #logger.info(f"Output Text: {output_text}")
        output_labels = self.decode_output(output_text)
        #logger.info(f"Output Labels: {output_labels}")

        output_dictionary["input_prompt"] = text_prompt
        output_dictionary["output_text"] = output_text
        output_dictionary["output_labels"] = output_labels

        return output_dictionary

    def forward_text(self, class_names, **kwargs):
        tok = self.processor.tokenizer
        emb = self.model.lm_head.weight
        letters = [self.current_class_to_lettre[c] for c in class_names]

        ids = [tok.encode(l, add_special_tokens=False) for l in letters]
        bad = [i for i, t in enumerate(ids) if len(t) != 1]
        if len(bad) != 0:
            logger.warning(f"We are in a class txt agregation scenarion we will aggregate the next {1} tokens for each class")
            
            x = torch.stack([emb.index_select(0, torch.tensor(t[:min(len(t), 1)], device=emb.device, dtype=torch.long)).mean(0) for t in ids], 0)
            return {"x_norm_txttoken": x}
            # OLD: raise ValueError(f"Non-single-token letters at idx {bad}: {[letters[i] for i in bad]} -> {ids}")

        else:
            token_ids = torch.tensor([t[0] for t in ids], device=emb.device, dtype=torch.long)
            x = emb.index_select(0, token_ids)  # [K, C]
            return {"x_norm_txttoken": x}

    def forward_text_heads(self, class_names = None):

        #K = len(class_names)
        device = next(self.model.parameters()).device  # or: self.model.lm_head.weight.device
        # Build Text Prompt
        # 1 format classes

        # 2 format chat
        class_prompts = [
            self.processor.apply_chat_template(
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": class_prompt},
                        ],
                    }
                ], 
                tokenize=False, 
                add_generation_prompt=True,
            ) for class_prompt in self.current_text_prompt_class
        ]

        # Build Input
        inputs = self.processor(text=class_prompts, padding=True, return_tensors="pt")
        inputs.to(device)


        # Get Output Hooks
        acts_llm_last_heads, remove_llm_last_heads = self.register_llm_attn_heads_last_token_hooks(self.model)
        acts_cls, remove_cls = self.register_llm_last_layer_prompt_lasttok_hook(self.model)

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=1)

        # Remove Hooks
        output_dictionary = {}
        output_dictionary["attn_heads"] = torch.stack([acts_llm_last_heads[i] for i in range(self.depth_llm)], dim=1)   
        remove_llm_last_heads()
        output_dictionary["x_norm_txttoken"] = acts_cls["x_norm_clstoken"]  # [B, 3584]
        remove_cls()        

        return output_dictionary


class LLaVAWrapper(VLVMWrapper):
    def __init__(self, model_name, weights, num_features_out, output_llm_lasttok_heads=False, **kwargs):
        super().__init__()
        self.type_model = "LMM"
        self.model_name = model_name

        #from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            weights, local_files_only=True, attn_implementation="sdpa", dtype=torch.float16, device_map="cuda"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id

        self.depth_llm = len(self.model.model.language_model.layers)
        self.num_features_hidden_llm = self.model.config.text_config.hidden_size
        self.num_features_out = num_features_out
        self.output_llm_lasttok_heads = output_llm_lasttok_heads

        def register_llm_last_layer_prompt_lasttok_hook(model):
            acts = {}
            layer = model.model.language_model.layers[-1]

            def f(module, inp, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                if hs.shape[1] > 1 and "x_norm_clstoken" not in acts:
                    acts["x_norm_clstoken"] = hs[:, -1, :].detach()

            h = layer.register_forward_hook(f)

            def remove():
                h.remove()

            return acts, remove

        def register_llm_attn_heads_last_token_hooks(model):
            acts = {}
            hooks = []
            layers = model.model.language_model.layers

            for i, layer in enumerate(layers):
                attn = layer.self_attn
                H = getattr(attn, "num_heads", attn.config.num_attention_heads)
                Dh = attn.head_dim

                def _pre_o_proj_hook(module, inp, i=i, H=H, Dh=Dh):
                    y = inp[0]  # [B, T, H*Dh] (post-attn, pre-o_proj)
                    if y.shape[1] > 1 and i not in acts:
                        acts[i] = y[:, -1, :].view(y.size(0), H, Dh).detach()

                hooks.append(attn.o_proj.register_forward_pre_hook(_pre_o_proj_hook))

            def remove():
                for h in hooks:
                    h.remove()

            return acts, remove

        self.register_llm_last_layer_prompt_lasttok_hook = register_llm_last_layer_prompt_lasttok_hook
        self.register_llm_attn_heads_last_token_hooks = register_llm_attn_heads_last_token_hooks
        self.need_training = None

    def forward(self, x: torch.Tensor, text_prompt=None):

        B = x.shape[0]
        device = x.device
        if text_prompt is None:
            text_prompt = self.current_text_prompt

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt},
            ],
        }]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        x_images = [[Image.fromarray((255 * x[b]).to(torch.uint8).permute(1, 2, 0).cpu().numpy())] for b in range(B)]
        inputs = self.processor(text=[text_prompt] * B, images=x_images, padding=True, return_tensors="pt")
        inputs.to(device)

        if self.output_llm_lasttok_heads:
            acts_heads, remove_heads = self.register_llm_attn_heads_last_token_hooks(self.model)
        acts_cls, remove_cls = self.register_llm_last_layer_prompt_lasttok_hook(self.model)

        _ = self.model.generate(**inputs, max_new_tokens=1)

        out = {"x_norm_clstoken": acts_cls["x_norm_clstoken"], "input_prompt": text_prompt}
        remove_cls()

        if self.output_llm_lasttok_heads:
            out["attn_heads"] = torch.stack([acts_heads[i] for i in range(self.depth_llm)], dim=1)
            remove_heads()

        return out

    def forward_text(self, class_names, **kwargs):
        tok = self.processor.tokenizer
        emb = self.model.lm_head.weight
        letters = [self.current_class_to_lettre[c] for c in class_names]
        ids = [tok.encode(l, add_special_tokens=False) for l in letters]
        token_ids = torch.tensor([t[0] for t in ids], device=emb.device, dtype=torch.long)
        x = emb.index_select(0, token_ids)
        return {"x_norm_txttoken": x}

    def forward_text_heads(self, class_names = None):
        device = next(self.model.parameters()).device

        class_prompts = [
            self.processor.apply_chat_template(
                [{
                    "role": "user",
                    "content": [{"type": "text", "text": class_prompt}],
                }],
                add_generation_prompt=True,
            )
            for class_prompt in self.current_text_prompt_class
        ]

        inputs = self.processor(text=class_prompts, padding=True, return_tensors="pt")
        inputs.to(device)

        acts_llm_last_heads, remove_llm_last_heads = self.register_llm_attn_heads_last_token_hooks(self.model)
        acts_cls, remove_cls = self.register_llm_last_layer_prompt_lasttok_hook(self.model)

        _ = self.model.generate(**inputs, max_new_tokens=1)

        output_dictionary = {}
        output_dictionary["attn_heads"] = torch.stack([acts_llm_last_heads[i] for i in range(self.depth_llm)], dim=1)
        remove_llm_last_heads()
        output_dictionary["x_norm_txttoken"] = acts_cls["x_norm_clstoken"]
        remove_cls()
        return output_dictionary

