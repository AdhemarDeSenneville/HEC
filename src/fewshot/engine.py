import torch
from torch import nn

# SET TRAINABLE FUNCTIONS

def set_all_trainable(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


class FinetuneWrapper(nn.Module):
    """
    Wraps a backbone and a classification head; applies freezing scheme per config.
    """
    def __init__(
            self,
            backbone: nn.Module,
            head: nn.Module,
            device,
        ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.device = device


    def init_peft(
            self,
            config,
        ):
        self.batch_size_max_infer = config['batch_size_max_infer']
        self.batch_size_max_train = config['batch_size_max_train']

        # -------- 
        #   HEAD   
        # --------
        mode_head = config['mode_head']
        set_all_trainable(self.head, False)
        
        if mode_head == "none":
            pass
        else:
            raise 'Error'

        # --------
        # BACKBONE 
        # --------
        mode_backbone = config['mode_backbone']
        set_all_trainable(self.backbone, False)

        if mode_backbone == "none":
            pass
        else:
            raise 'Error'

        self.frozen_backbone = True
        for p in self.backbone.parameters():
            if p.requires_grad:
                self.frozen_backbone = False
                break
        
        self.frozen_head = True
        for p in self.head.parameters():
            if p.requires_grad:
                self.frozen_head = False
                break
        
        self.frozen_model = self.frozen_backbone and self.frozen_head


    def forward(self, x: torch.Tensor, **kwargs) -> dict:

        # Backbone Inference
        if self.frozen_backbone: # WARNING: Test this code 
            with torch.no_grad():
                x_backbone_dict = self.backbone(x, **kwargs)
        else:
            x_backbone_dict = self.backbone(x)
        
        # Head Inference
        if self.frozen_head and self.frozen_backbone:
            with torch.no_grad():
                x_head_dict = self.head(x_backbone_dict)
        else:
            x_head_dict = self.head(x_backbone_dict)

        return {
            **x_backbone_dict,
            **x_head_dict,
        }
    

    def infer(self, x: torch.Tensor, **kwargs) -> dict:
        """
        Chunked forward pass: split x into batches of size self.batch_infer_max,
        run forward, then concat tensor outputs along dim=0.
        """
        temp_mode = self.training # WARNING test the impact
        self.eval()
        max_b = self.batch_size_max_infer
        if max_b <= 0 or x.size(0) <= max_b:
            with torch.no_grad():
                merged = self.forward(x)
        else:
            with torch.no_grad():
                cat_buffers = None
                static_buf = {}
                N = x.size(0)
                for start in range(0, N, max_b):
                    out = self.forward(x[start:start + max_b], **kwargs)  # dict(k -> tensor)
                    if cat_buffers is None:
                        cat_buffers = {k: [v] for k, v in out.items() if torch.is_tensor(v)}
                        static_buf = {k: v for k, v in out.items() if not torch.is_tensor(v)}
                    else:
                        for k, v in out.items():
                            if torch.is_tensor(v):
                                cat_buffers[k].append(v)
                merged = {k: torch.cat(vs, dim=0) for k, vs in cat_buffers.items()}
                merged.update(static_buf)
        
        if temp_mode:
            self.train()
            #print("BACK TO TRAINING MODE", self.training)
            
        return merged


    def get_num_train_params(self) -> dict:
        train_params_backbone = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        train_params_head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {
            "train_params_backbone": train_params_backbone,
            "train_params_head": train_params_head,
            "train_params": train_params_backbone + train_params_head,
        }

