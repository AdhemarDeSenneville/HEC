import os, sys
import logging, warnings

import torch
import torch.nn as nn
import numpy as np

from itertools import product
from omegaconf import OmegaConf

# None head and transform (We use no head and no augmentations)
class Head(nn.Module):
    """
    Base class for all heads.
    """
    def __init__(self):
        super().__init__()
        self.is_reprojection = False
    
    def count_uncollapse(self):
        return {'collapse_count': -1}

class NoneHead(Head):
    """
    MLP head: prend x['x_norm_clstoken'] -> logits, probs.
    Même interface que LinearHead, sans normalisation.
    """
    def __init__(
            self,
            input_mode,
            num_features_in,
            **kwargs,
        ):
        self.input_mode = input_mode
        self.num_features_out = num_features_in
        super().__init__()


    def forward(self, x):
        return {
            "x_head": get_proto(x, self.input_mode, detach=False)
        }

def build_none_transform(**kwargs):
    return None


# Get Prototypes
def get_proto(features, mode, detach = True, temperature = 0.1):
    if mode == 'cls':
        x = features['x_norm_clstoken']                  # [B, C]
    else:
        raise NotImplementedError(f"Mode {mode} not implemented for get_proto()")

    if detach:
        return x.detach() 
    else:
        return x


# NCC Stuff
def get_NCC_accuracy(
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor, 
        query_labels: torch.Tensor
    ) -> float:
    Xs = support_features.float()
    ys = support_labels.long()
    Xq = query_features.float()
    yq = query_labels.long()

    Xs = Xs / (Xs.norm(dim=1, keepdim=True) + 1e-12)
    Xq = Xq / (Xq.norm(dim=1, keepdim=True) + 1e-12)

    classes = torch.unique(ys)
    protos = []
    for c in classes:
        p = Xs[ys == c].mean(dim=0)
        protos.append(p / (p.norm() + 1e-12))
    P = torch.stack(protos, dim=0)            # [K, D]

    sims = Xq @ P.T                            # [Nq, K]
    preds = classes[sims.argmax(dim=1)]
    return float((preds == yq).float().mean().item())

def get_NCC_labels(
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor
    ) -> np.ndarray:
    Xs = support_features.float()
    ys = support_labels.long()
    Xq = query_features.float()

    Xs = Xs / (Xs.norm(dim=1, keepdim=True) + 1e-12)
    Xq = Xq / (Xq.norm(dim=1, keepdim=True) + 1e-12)

    classes = torch.unique(ys)
    protos = []
    for c in classes:
        p = Xs[ys == c].mean(dim=0)
        protos.append(p / (p.norm() + 1e-12))
    P = torch.stack(protos, dim=0)            # [K, D]

    sims = Xq @ P.T                            # [Nq, K]
    preds = classes[sims.argmax(dim=1)]
    return preds.detach().cpu().numpy().astype(np.int64)


# Utils Head Selection
def get_top_heads_T(
        dataset_name,
        mode,
        model_name,
        encoding,
    ):

    if encoding == "direct":
        if model_name == "QWENv2":
            top_head_dict = {
                "oxford_pets": [640, 655, 725, 617, 661, 695, 772, 641, 699, 642, 633, 750, 623, 754, 650, 656, 697, 628, 755, 693, 619, 629, 752, 631, 651, 646, 643, 742, 744, 751],
                "eurosat": [593, 655, 750, 699, 752, 631, 725, 641, 695, 754, 751, 623, 744, 629, 590, 753, 755, 780, 642, 587, 612, 693, 690, 617, 549, 697, 583, 554, 692, 691],
                "ucf101": [593, 633, 628, 655, 755, 699, 754, 695, 592, 631, 752, 750, 634, 623, 697, 640, 725, 549, 644, 629, 646, 612, 647, 641, 690, 590, 693, 617, 538, 642],
                "sun397": [593, 628, 631, 754, 592, 750, 752, 661, 643, 755, 633, 697, 690, 772, 634, 612, 640, 693, 751, 655, 775, 647, 695, 642, 744, 725, 699, 691, 587, 549],
                "caltech101": [655, 623, 628, 699, 617, 661, 693, 629, 752, 640, 695, 593, 631, 642, 755, 641, 725, 592, 750, 772, 633, 754, 751, 643, 690, 612, 697, 619, 587, 744],
                "dtd": [693, 593, 699, 590, 640, 623, 750, 655, 641, 633, 628, 661, 617, 650, 754, 752, 725, 642, 629, 646, 631, 755, 587, 612, 634, 644, 549, 726, 647, 695],
                "fgvc": [695, 661, 750, 631, 655, 628, 642, 650, 754, 725, 640, 633, 752, 751, 755, 651, 629, 641, 699, 693, 647, 656, 649, 617, 753, 644, 726, 723, 648, 643],
                "food101": [628, 655, 631, 650, 661, 633, 695, 644, 755, 592, 549, 623, 751, 640, 612, 617, 629, 752, 642, 593, 725, 646, 697, 647, 750, 754, 645, 693, 699, 587],
                "oxford_flowers": [655, 725, 633, 642, 695, 661, 628, 640, 617, 697, 699, 631, 650, 651, 641, 623, 751, 755, 693, 656, 752, 629, 754, 772, 750, 744, 549, 619, 646, 643],
                "stanford_cars": [647, 650, 633, 631, 697, 644, 634, 649, 648, 772, 538, 651, 628, 743, 698, 549, 612, 592, 686, 593, 583, 689, 759, 661, 728, 775, 724, 587, 734, 729],
                "ALL": [631, 628, 633, 655, 750, 697, 752, 640, 650, 661, 755, 593, 695, 699, 725, 754, 642, 629, 693, 623, 641, 644, 617, 772, 647, 751, 612, 549, 592, 587]
            }
        elif model_name == "LLaVA_OV":
            top_head_dict = {
                "oxford_pets": [650, 655, 725, 623, 640, 628, 754, 750, 633, 755, 695, 775, 697, 617, 661, 742, 752, 648, 642, 641, 772, 656, 631, 635, 744, 629, 647, 699, 644, 762],
                "eurosat": [631, 699, 691, 593, 655, 695, 725, 687, 549, 587, 754, 602, 583, 744, 629, 628, 641, 617, 743, 642, 742, 575, 554, 556, 644, 750, 582, 581, 643, 690],
                "ucf101": [750, 754, 644, 655, 633, 628, 631, 593, 752, 646, 617, 775, 755, 640, 725, 641, 612, 634, 623, 649, 647, 699, 691, 698, 592, 642, 697, 650, 587, 656],
                "sun397": [631, 754, 750, 628, 640, 691, 612, 644, 693, 593, 623, 687, 751, 697, 661, 587, 725, 634, 655, 641, 695, 633, 775, 583, 753, 642, 602, 755, 699, 752],
                "caltech101": [628, 655, 617, 750, 629, 641, 699, 631, 640, 661, 593, 602, 754, 725, 623, 644, 633, 583, 775, 693, 697, 549, 587, 744, 570, 577, 592, 695, 753, 643],
                "dtd": [699, 593, 641, 628, 623, 754, 655, 590, 693, 617, 633, 750, 646, 661, 629, 640, 726, 631, 725, 592, 612, 602, 752, 642, 644, 650, 772, 691, 657, 775],
                "fgvc": [754, 750, 633, 631, 725, 655, 642, 640, 650, 641, 698, 644, 647, 753, 697, 752, 695, 649, 742, 656, 726, 661, 628, 635, 775, 755, 648, 643, 734, 617],
                "food101": [628, 644, 631, 650, 633, 647, 661, 750, 640, 646, 655, 697, 754, 641, 725, 549, 629, 642, 593, 623, 617, 775, 612, 752, 645, 587, 592, 583, 695, 656],
                "oxford_flowers": [750, 695, 628, 697, 725, 661, 655, 754, 617, 642, 775, 646, 640, 699, 593, 650, 633, 641, 644, 693, 623, 629, 744, 602, 752, 631, 755, 583, 549, 656],
                "stanford_cars": [633, 647, 648, 661, 631, 697, 628, 650, 752, 644, 775, 725, 743, 649, 698, 695, 728, 750, 593, 651, 755, 744, 753, 583, 686, 693, 742, 640, 629, 771],
                "ALL": [750, 628, 631, 754, 725, 633, 640, 644, 661, 593, 697, 655, 775, 695, 617, 699, 650, 752, 647, 641, 629, 623, 693, 642, 744, 583, 698, 755, 602, 549]
            }
        elif model_name == "IDEFICSv2":
            top_head_dict = {
                "oxford_pets": [896, 1012, 624, 908, 662, 909, 617, 616, 930, 911, 606, 1014, 670, 1015, 637, 1008, 619, 829, 910, 626, 940, 942, 660, 661, 789, 941, 627, 733, 791, 810],
                "eurosat": [1017, 1019, 1014, 613, 588, 1012, 942, 940, 908, 911, 909, 896, 614, 930, 811, 941, 944, 617, 970, 692, 624, 999, 1015, 857, 657, 663, 487, 928, 733, 662],
                "ucf101": [588, 615, 1012, 911, 908, 692, 801, 930, 695, 989, 896, 909, 940, 1019, 936, 657, 613, 1014, 693, 662, 606, 941, 942, 1015, 1018, 928, 882, 772, 1008, 815],
                "sun397": [692, 588, 615, 1019, 1017, 1012, 693, 911, 1018, 936, 617, 930, 662, 695, 909, 657, 896, 941, 801, 614, 989, 1014, 772, 940, 624, 908, 815, 606, 910, 1015],
                "caltech101": [911, 1012, 896, 908, 1019, 930, 936, 615, 617, 588, 1017, 1015, 662, 1014, 909, 910, 624, 989, 616, 692, 606, 801, 941, 1018, 928, 988, 693, 772, 733, 829],
                "dtd": [617, 1012, 940, 624, 606, 615, 789, 941, 588, 1015, 657, 896, 911, 616, 662, 988, 1014, 908, 989, 810, 613, 942, 670, 930, 790, 614, 809, 660, 692, 1019],
                "fgvc": [909, 617, 908, 989, 1012, 662, 588, 911, 638, 624, 930, 1014, 661, 988, 615, 942, 626, 606, 789, 733, 1019, 896, 1015, 810, 637, 1008, 936, 627, 616, 815],
                "food101": [911, 908, 989, 930, 615, 1012, 940, 909, 936, 588, 662, 692, 772, 606, 815, 789, 941, 1019, 896, 661, 1014, 660, 1015, 988, 617, 943, 1017, 809, 801, 733],
                "oxford_flowers": [1012, 908, 930, 911, 662, 909, 942, 617, 896, 1015, 615, 606, 588, 989, 1014, 616, 661, 624, 1019, 940, 733, 660, 637, 652, 619, 969, 936, 941, 1017, 789],
                "stanford_cars": [588, 789, 911, 637, 930, 1012, 908, 928, 791, 809, 615, 940, 790, 638, 1019, 657, 921, 909, 936, 652, 606, 780, 941, 1008, 772, 1018, 942, 943, 670, 997],
                "ALL": [1012, 911, 908, 588, 930, 615, 909, 896, 1019, 617, 1014, 940, 606, 1015, 662, 989, 941, 624, 936, 789, 1017, 637, 928, 942, 910, 657, 619, 988, 733, 772]
            }
        elif model_name == "FINEDEFICS":
            top_head_dict = {
                "oxford_pets": [617, 619, 616, 896, 910, 1014, 606, 624, 590, 909, 989, 661, 940, 930, 969, 637, 1005, 588, 1012, 941, 652, 1016, 638, 971, 626, 644, 662, 789, 487, 670],
                "eurosat": [487, 896, 458, 1012, 481, 616, 1014, 523, 544, 969, 971, 614, 989, 522, 588, 541, 859, 1019, 606, 1017, 584, 589, 622, 1008, 512, 733, 617, 474, 547, 568],
                "ucf101": [588, 606, 481, 615, 613, 657, 544, 616, 589, 617, 624, 1014, 940, 519, 941, 666, 928, 487, 568, 638, 930, 896, 733, 1012, 614, 591, 1017, 511, 692, 695],
                "sun397": [481, 613, 617, 589, 1017, 588, 1016, 624, 615, 693, 941, 519, 616, 666, 560, 1014, 544, 569, 695, 733, 657, 591, 896, 590, 638, 568, 606, 511, 940, 487],
                "caltech101": [617, 1016, 989, 896, 1014, 616, 588, 941, 613, 930, 615, 910, 481, 969, 657, 638, 1012, 589, 606, 1017, 971, 928, 624, 940, 619, 666, 909, 590, 1005, 733],
                "dtd": [588, 617, 589, 615, 1014, 616, 941, 910, 657, 613, 614, 1012, 940, 606, 989, 896, 481, 624, 969, 544, 1016, 666, 678, 692, 590, 569, 560, 619, 971, 638],
                "fgvc": [619, 617, 652, 624, 910, 908, 626, 606, 1014, 616, 989, 733, 654, 930, 661, 1012, 829, 969, 662, 988, 971, 928, 909, 896, 627, 670, 789, 678, 1008, 941],
                "food101": [617, 588, 941, 930, 940, 616, 481, 589, 606, 619, 613, 624, 989, 615, 1014, 638, 896, 1016, 910, 733, 909, 544, 1012, 936, 1017, 614, 670, 969, 657, 568],
                "oxford_flowers": [619, 624, 617, 940, 941, 616, 896, 606, 626, 1014, 644, 989, 670, 910, 930, 1012, 661, 588, 652, 1016, 678, 969, 733, 1005, 654, 971, 590, 662, 728, 909],
                "stanford_cars": [638, 637, 789, 911, 790, 481, 791, 487, 936, 809, 1017, 829, 588, 1014, 619, 989, 909, 626, 1012, 1019, 930, 1008, 772, 679, 519, 590, 941, 568, 928, 908],
                "ALL": [617, 1014, 588, 616, 606, 896, 941, 624, 989, 481, 619, 1012, 930, 910, 638, 940, 589, 969, 487, 909, 626, 657, 1017, 615, 670, 971, 613, 928, 661, 568]
            }
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    if mode == "Task":
        top_heads = top_head_dict["ALL"]

    if mode == "Domain":
        top_heads = top_head_dict[dataset_name]
    
    return top_heads

def get_top_heads_V(
        dataset_name,
        mode,
        model_name = "QWENv2",
    ):

    if model_name == "QWENv2":
        top_head_dict = {
            "oxford_pets": [781, 628, 753, 755, 651, 749, 783, 647, 754, 756, 750, 748, 751, 701, 752, 693, 767, 763, 704, 766, 768, 769, 736, 650, 776, 718, 742, 720, 764, 706],
            "eurosat": [660, 556, 659, 485, 421, 692, 631, 594, 627, 592, 472, 669, 471, 688, 611, 433, 717, 703, 652, 423, 690, 651, 427, 420, 656, 504, 720, 532, 436, 684],
            "ucf101": [594, 765, 538, 660, 632, 748, 701, 692, 683, 686, 764, 646, 574, 781, 749, 611, 703, 783, 625, 652, 575, 690, 706, 751, 653, 752, 644, 738, 733, 593],
            "sun397": [754, 783, 764, 749, 775, 758, 753, 750, 771, 728, 449, 601, 698, 742, 690, 751, 767, 766, 768, 752, 769, 602, 781, 582, 704, 763, 736, 735, 760, 776],
            "caltech101": [736, 749, 771, 775, 750, 764, 763, 767, 768, 769, 755, 766, 752, 754, 783, 714, 748, 699, 695, 715, 701, 762, 720, 706, 753, 742, 718, 733, 704, 781],
            "dtd": [749, 698, 755, 753, 704, 783, 771, 751, 752, 750, 736, 754, 640, 690, 727, 693, 734, 643, 742, 758, 695, 776, 775, 735, 743, 762, 718, 646, 744, 714],
            "fgvc": [749, 647, 644, 660, 628, 781, 753, 594, 755, 690, 562, 611, 748, 651, 758, 574, 783, 782, 756, 727, 708, 722, 669, 706, 721, 658, 554, 649, 720, 751],
            "food101": [758, 752, 783, 753, 754, 751, 749, 750, 771, 743, 734, 695, 725, 650, 706, 742, 755, 699, 643, 711, 744, 773, 733, 720, 704, 715, 762, 760, 693, 640],
            "oxford_flowers": [756, 748, 650, 751, 781, 749, 686, 628, 721, 758, 633, 646, 765, 644, 783, 660, 761, 594, 727, 690, 625, 723, 722, 611, 733, 455, 652, 776, 689, 757],
            "stanford_cars": [783, 781, 758, 749, 782, 755, 753, 748, 751, 764, 767, 768, 766, 763, 723, 769, 603, 712, 752, 744, 736, 633, 742, 714, 704, 720, 718, 750, 715, 562],
            "ALL": [781, 690, 594, 751, 783, 758, 611, 720, 701, 749, 706, 660, 727, 704, 756, 721, 736, 755, 748, 703, 753, 718, 715, 764, 771, 628, 659, 767, 776, 769],
        }

        if mode == "Task":
            top_heads = top_head_dict["ALL"]

        if mode == "Domain":
            top_heads = top_head_dict[dataset_name]
        
        return top_heads

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


# Utils Hparams Search
def instantiate_grid_search(cfg):
    if "sweep_cfg" not in cfg or cfg.sweep_cfg is None:
        return []

    base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    del base["sweep_cfg"]

    keys = list(cfg.sweep_cfg.keys())
    values = [list(cfg.sweep_cfg[k]) for k in keys]

    out = []
    for combo in product(*values):
        c = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        for k, v in zip(keys, combo):
            c[k] = v
        out.append(c)
    return out


# Utils TensorFlow (Suppress Warnings)
def tg_tf():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["AUTOGRAPH_VERBOSITY"] = "0"
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"tensorflow(\.|$)")
    warnings.filterwarnings("ignore", category=FutureWarning,      module=r"tensorflow(\.|$)")
    warnings.filterwarnings("ignore", message=r".*use fn_output_signature instead.*")
    warnings.filterwarnings("ignore", message=r".*choose_from_datasets_v2.*")
    warnings.filterwarnings("ignore", message=r".*Executor start aborting.*")
    warnings.filterwarnings("ignore", message=r".*replicate on split optimization.*")
    warnings.filterwarnings("ignore", message=r".*Cannot dlopen some GPU libraries.*")
    warnings.filterwarnings("ignore", category=FutureWarning,
                            message=r".*You are using `torch\.load` with `weights_only=False`.*")

    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")

    for name in ("tensorflow", "tensorflow.experimental", "tensorflow.python", "absl", "absl.logging"):
        lg = logging.getLogger(name)
        lg.propagate = False
        lg.setLevel(logging.ERROR)
        for h in list(lg.handlers):
            lg.removeHandler(h)

    if "tensorflow" in sys.modules:
        import tensorflow as tf
        tf.get_logger().handlers[:] = []
        tf.get_logger().propagate = False
        tf.get_logger().setLevel("ERROR")
        tf.autograph.set_verbosity(0)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


