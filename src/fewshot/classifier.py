import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from .utils import(
    get_proto, 
    get_NCC_accuracy, 
    get_NCC_labels,
    get_top_heads_V,
    get_top_heads_T,
)

from .prompt import get_prompt_clip

import logging
logger = logging.getLogger(__name__)

def make_grad_delay(module, n_steps: int):
    if n_steps == 0:
        return lambda: None, lambda: None
    state = {"t": 0}
    handles = []

    def hook(grad):
        return grad * 0 if state["t"] < n_steps else grad

    for p in module.parameters():
        if p.requires_grad:                 # works with PEFT: only trainable params get hooks
            handles.append(p.register_hook(hook))

    def step():   state["t"] += 1          # call once AFTER each loss.backward()
    def remove(): [h.remove() for h in handles]

    return step, remove


class Classifier:
    """
    A baseline classifier using L2 distance to class prototypes.
    """


    def __init__(
            self, 
            input_mode: str = 'head',
            transforms = None,
        ):
        self.input_mode = input_mode
        self.transforms = transforms
        self.cached_frozen_features = False


    def log(self, model, loss_fn=None):


        def _stats(mod):
            abs_vals, abs_grads = [], []
            for p in mod.parameters():
                if not p.requires_grad:
                    continue
                with torch.no_grad():
                    abs_vals.append(p.data.abs().mean())
                    if p.grad is not None:
                        abs_grads.append(p.grad.data.abs().mean())
            param_mean = torch.stack(abs_vals).mean().item() if abs_vals else float("nan")
            grad_mean  = torch.stack(abs_grads).mean().item() if abs_grads else float("nan")
            return param_mean, grad_mean
        # per-part
        bb_param, bb_grad = _stats(model.backbone) if hasattr(model, "backbone") else (float("nan"), float("nan"))
        hd_param, hd_grad = _stats(model.head)     if hasattr(model, "head")     else (float("nan"), float("nan"))
        mdl_param, mdl_grad = _stats(model)

        log_dict = {
            #"accuracy": accuracy,
            "param_backbone": bb_param,
            "grad_backbone":  bb_grad,
            "param_head":     hd_param,
            "grad_head":      hd_grad,
            "param_model":    mdl_param,
            "grad_model":     mdl_grad,
        }
        
        try:
            log_dict['param_loss'], log_dict['grad_loss'] = _stats(loss_fn) 
        except:
            pass
        return log_dict


    def fit_pred(self, model, support_set, support_label, query_set, query_label):

        self.num_features_in = model.head.num_features_out

        self.does_inference = False
        if self.transforms:
            self.does_inference = True

        if not model.frozen_model:
            self.does_inference = True

        if not self.does_inference:
            with torch.no_grad():
                self.support_set_features = model.infer(support_set)
                self.query_set_features   = model.infer(query_set)

        self.query_label_for_log = query_label
        self.query_set_for_log = query_set

        self.fit(model, support_set, support_label, query_set)
        return self.pred(model, support_set, support_label, query_set, query_label)


    def cache_frozen_features(self, support_set_features, query_set_features):
        self.support_set_features = support_set_features
        self.query_set_features = query_set_features
        self.cached_frozen_features = True


    def get_frozen_features(self, model, support_set, query_set):

        assert self.frozen_classifier, "Classifier must be frozen to get frozen features."

        if self.cached_frozen_features:
            return self.support_set_features, self.query_set_features
        else:
            with torch.no_grad():
                support_set_features = model.infer(support_set)
                query_set_features   = model.infer(query_set)
            return support_set_features, query_set_features


    def init_task_specific_data(self, label_to_str, dataset_name, benchmark_name):
        self.current_label_to_str = label_to_str
        self.current_dataset_name = dataset_name
        self.current_benchmark_name = benchmark_name


# ------------------------------------------- #
# -----------   Vision Frozen   ------------- #
# ------------------------------------------- #


class ClassifierBaselineSimple(Classifier):
    frozen_classifier = True
    
    def __init__(
            self, 
            input_mode='cls', 
            fit_mode='linear_CE', 
            weight_decay=0.0,
            fit_hyperparams={},
            transforms=None,
            log_losses: bool = False,
            ):
        assert fit_mode in {'linear_CE', 'KNN', 'linear_SVM', 'linear_ridge'}
        self.fit_mode = fit_mode
        self.weight_decay = weight_decay
        self.fit_hyperparams = fit_hyperparams
        self.model = None
        self.unique_labels = None
        super().__init__(
            input_mode= input_mode,
            transforms = transforms,
        )


    def fit_pred(self, model, support_set, support_label, query_set, query_label):

        #logger.info("Inference:")
        support_set_features, query_set_features = self.get_frozen_features(model, support_set, query_set)
        ##logger.info("Fitting:")

        x_s_t = get_proto(support_set_features, self.input_mode)
        x_q_t = get_proto(query_set_features, self.input_mode)

        device, dtype = x_q_t.device, x_q_t.dtype
        Xs = x_s_t.detach().cpu().numpy().astype(np.float64)
        Xq = x_q_t.detach().cpu().numpy().astype(np.float64)

        # map labels to 0..K-1 with fixed ordering
        self.unique_labels = torch.unique(support_label).sort()[0]
        y_idx = torch.empty_like(support_label, dtype=torch.long)
        for i, lbl in enumerate(self.unique_labels):
            y_idx[support_label == lbl] = i
        ys = y_idx.detach().cpu().numpy()
        K = len(self.unique_labels)

        if self.fit_mode == 'linear_ridge':
            Ys = np.zeros((Xs.shape[0], K), dtype=np.float64)
            Ys[np.arange(Xs.shape[0]), ys] = 1.0

            self.model = Ridge(alpha=self.weight_decay, fit_intercept=self.fit_hyperparams['fit_intercept'])
            self.model.fit(Xs, Ys)
            logits_np = self.model.predict(Xq)   # [Nq, K]
        else:
            raise ValueError(f"Unknown fit_mode: {self.fit_mode}")

        logits = torch.from_numpy(np.atleast_2d(logits_np)).to(device=device, dtype=dtype)
        probs = F.softmax(logits, dim=1)
        scores, top_idx = probs.max(dim=1)

        query_label_pred = self.unique_labels.to(top_idx.device)[top_idx]
        accuracy = (query_label_pred == query_label).float().mean().item()



        #logger.info("Rest of the stuff:")

        return {
            "query_label_pred": query_label_pred,
            "query_label_pred_score": scores,
            "logits": logits,
            "accuracy": accuracy,
        }


    def pred_info(self, features):
        x_t = get_proto(features, self.input_mode)
        X = x_t.detach().cpu().numpy().astype(np.float64)
        device, dtype = x_t.device, x_t.dtype

        if self.fit_mode == 'linear':
            logits_np = X @ self.model.coef_.T + self.model.intercept_
        else:
            probs_np = self.model.predict_proba(X)
            logits_np = np.log(probs_np + 1e-12)

        logits = torch.from_numpy(np.atleast_2d(logits_np)).to(device=device, dtype=dtype)
        return {
            "logits": logits,
            "logits_average": logits,
        }


class ClassifierSAV(Classifier):
    # Reimplemntation from https://github.com/chancharikmitra/SAVs
    frozen_classifier = True
    
    def __init__(
        self,
        input_mode = None,
        transforms = None,
        log_losses: bool = False,
        k: int = 20,
        rank_score: dict = {"type": "NCC_accuracy"},
        pred_method: dict = {"type": "NCC"},
        ensemble_method = "majority_vote",
    ):
        super().__init__(
            input_mode=input_mode,
            transforms=transforms,
        )
        self.k = k
        self.rank_score = rank_score
        self.ensemble_method = ensemble_method
        self.pred_method = pred_method
    
    def get_rank_score(self, features, labels):

        if self.rank_score['type'] == 'NCC_accuracy':
            return get_NCC_accuracy(
                features, labels, features, labels
            )
        else:
            raise ValueError(self.rank_score['type'])
    
    def get_pred_labels(self, support_features, support_labels, query_features):
        
        if self.pred_method['type'] == 'NCC':
            return get_NCC_labels(
                support_features, support_labels, query_features
            )
        else:
            raise ValueError(self.pred_method)

    def ensemble(self, votes, weights):

        if self.ensemble_method == "majority_vote":
            def majority_vote(votes):
                out = np.zeros(votes.shape[1], dtype=np.int64)
                for i in range(votes.shape[1]):
                    out[i] = np.bincount(votes[:, i]).argmax()
                return out
            return majority_vote(votes)
        else:
            raise ValueError(self.ensemble_method)

    def fit(self, support_set_features: torch.Tensor, support_label: torch.Tensor):
        
        A = support_set_features["attn_heads"].float() # (Ns, L, H, Dh)
        
        Ns, L, H, Dh = A.shape
        M = L * H
        A = A.reshape(Ns, M, Dh)
        A = F.normalize(A, dim=-1) 

        head_scores = torch.empty(M, device=A.device, dtype=torch.float32)
        for m in range(M):
            head_scores[m] = float(self.get_rank_score(A[:, m, :], support_label))

        order = torch.argsort(head_scores, descending=True)
        self.selected_idx = order[: min(self.k, M)]
        self.selected_w = head_scores[self.selected_idx].detach().cpu().numpy().astype(np.float32)

        self.support_A = A.detach()
        self.support_label = support_label.detach()
        self.L = L
        self.H = H
        return self

    def pred(self, query_set_features: torch.Tensor, query_label: torch.Tensor):
        Aq = query_set_features["attn_heads"].float() # (Nq, L, H, Dh)
        Nq, L, H, Dh = Aq.shape
        Aq = Aq.reshape(Nq, L * H, Dh)
        Aq = F.normalize(Aq, dim=-1)

        votes_list = []
        for m in self.selected_idx.tolist():
            sf = self.support_A[:, m, :]
            qf = Aq[:, m, :]
            votes_list.append(self.get_pred_labels(sf, self.support_label, qf))

        votes = np.stack(votes_list, axis=0)  # (k, Nq)
        yhat = self.ensemble(votes, self.selected_w)
        yq = query_label.detach().cpu().numpy().astype(np.int64)
        accuracy = float((yhat == yq).mean())

        return {"accuracy": accuracy, "pred": yhat, "top_idx": self.selected_idx.detach().cpu().numpy()}

    def fit_pred(self, model, support_set, support_label, query_set, query_label):
        
        support_set_features, query_set_features = self.get_frozen_features(model, support_set, query_set)

        self.fit(support_set_features, support_label)
        return self.pred(query_set_features, query_label)


class ClassifierHEC_V(Classifier):
    frozen_classifier = True
    
    def __init__(
        self,
        input_mode = None,
        transforms = None,
        log_losses: bool = False,
        k: int = 20,
        tau: float = 10.0,
        cov_reg: float = 1.0,
        ensemble_method = "mean_prob",
        context = "Class", # "Class" or "Task" or "Domain"

        ensemble_param = 1,
    ):
        super().__init__(
            input_mode=input_mode,
            transforms=transforms,
        )
        self.k = k
        self.tau = tau
        self.cov_reg = cov_reg
        self.ensemble_method = ensemble_method
        self.ensemble_param = ensemble_param

        if context in {"Task", "Domain"} and log_losses:
            raise ValueError("Cannot log losses when context is Task or Domain, because there is only one head and no training is done.")

        self.context = context
        self.log_losses = log_losses


    
    def get_rank_scores(
            self, 
            support_features, # [B, L*H, C]
            support_labels,   # [B]
        ):
        # Pre Prrocessing
        N = support_features.shape[0]
        M = support_features.shape[1]   
        C = support_features.shape[2]
        labels_unique = torch.unique(support_labels)
        y_idx = torch.searchsorted(labels_unique, support_labels)  # [B]

        # FIT THE MODEL
        
        # Compute Class Prototypes
        support_features = F.normalize(support_features, p=2, dim=-1)                                         # [B, L*H, C]
        class_mean = torch.stack([support_features[support_labels == lbl].mean(0) for lbl in labels_unique])  # [K, L*H, C] 

        # Compute Covariance Matrix
        support_features_centred =  (support_features - class_mean[y_idx]).transpose(0, 1)     # [L*H, B, C]
        cov = (support_features_centred.transpose(1, 2) @ support_features_centred) / (N - 1)  # [L*H, C, C]
        tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
        cov_inv = C * torch.linalg.pinv(                                                       # [L*H, C, C]
            (N - 1) * cov + \
            self.cov_reg * tr[:, None, None] * \
            torch.eye(C, device=support_features_centred.device, dtype=support_features_centred.dtype)[None, :, :]
        )

        # EVALUATE THE MODEL

        # Optimized algorithm: avoids materializing [Nq, topk, K, C].
        mahal = (
            torch.einsum('nld,ldf,nlf->nl', support_features, cov_inv, support_features)[:, :, None]
            - 2.0 * torch.einsum('nld,ldf,lkf->nlk', support_features, cov_inv, class_mean.permute(1, 0, 2))
            + torch.einsum('lkd,ldf,lkf->lk', class_mean.permute(1, 0, 2), cov_inv, class_mean.permute(1, 0, 2))[None, :, :]
        )  # [B, L*H, K]

        # Mainly for the (not tested) case of class imbalance:
        counts = torch.stack([(support_labels == lbl).sum() for lbl in labels_unique]).to(support_features.dtype)
        log_prior = (counts / counts.sum()).log()        # [K]

        logits = -0.5 * mahal + log_prior[None, None, :] # [B, L*H, K]
        z = torch.softmax(logits / self.tau, dim=-1)     # [B, L*H, K]
        head_scores = z[torch.arange(support_labels.numel(), device=support_labels.device), :, y_idx].mean(dim=0)  # [L*H] # [L*H]
        return {
            "head_scores": head_scores,      # [L*H]
            "head_predictions": z,           # [B, L*H, K]
            "head_logits": logits,           # [B, L*H, K]
            "head_means": class_mean.permute(1, 0, 2), # [L*H, K, C]
            "head_cov_inv": cov_inv,         # [L*H, C, C]
            "log_prior": log_prior,          # [K]
            "labels_unique": labels_unique,  # [K]
        }

    def get_pred_labels(
            self, 
            query_features, # [Nq, topk, C]
            head_means,     # [topk, K, C]
            head_cov_inv,   # [topk, C, C]
            log_prior,      # [K]
        ):
        query_features = F.normalize(query_features, p=2, dim=-1)  # [Nq, topk, C]

        # Optimized algorithm: avoids materializing [Nq, topk, K, C].
        mahal = (
            torch.einsum('ntd,tdf,ntf->nt', query_features, head_cov_inv, query_features)[:, :, None]
            - 2.0 * torch.einsum('ntd,tdf,tkf->ntk', query_features, head_cov_inv, head_means)
            + torch.einsum('tkd,tdf,tkf->tk', head_means, head_cov_inv, head_means)[None, :, :]
        )  # [Nq, topk, K]
            
                                                       
        logits = -0.5 * mahal + log_prior[None, None, :]  # [Nq, topk, K]
        z = torch.softmax(logits / self.tau, dim=-1)      # [Nq, topk, K]
        return z, logits       # [Nq, topk, K]

    def get_heads_accuracy(
            self,
            query_features, # [Nq, topk, C]
            query_labels,   # [Nq]
            head_means,     # [topk, K, C]
            head_cov_inv,   # [topk, C, C]
            log_prior,      # [K]  
            labels_unique,
        ):
        z, _ = self.get_pred_labels(
            query_features,
            head_means,
            head_cov_inv,
            log_prior,
        )  # [Nq, topk, K]

        y_hat = z.argmax(dim=-1)                                                       # [Nq, topk]
        y_idx = torch.searchsorted(labels_unique, query_labels)                        # [Nq]

        correct = (y_hat == y_idx[:, None]).to(z.dtype)                                # [Nq, topk]
        onehot = F.one_hot(y_idx, num_classes=labels_unique.numel()).to(z.dtype)       # [Nq, K]

        acc_num = torch.einsum('nt,nk->tk', correct, onehot)                           # [topk, K]
        acc_den = onehot.sum(dim=0)                                                    # [K]
        return acc_num / acc_den[None, :]                                              # [topk, K]

    def ensemble(
            self,
            head_scores,         # [M]
            query_predictions,   # [Nq, M, K]
            query_logits,        # [Nq, M, K]
            support_predictions, # [Ns, M, K] # ONLY for some methods
            support_logits,      # [Ns, M, K] # ONLY for some methods
            support_labels,      # [Ns]       # ONLY for some methods
        ):

        # PROB ENSEMBLE
        if self.ensemble_method == "mean_prob":
            return query_predictions.mean(dim=1).argmax(dim=-1).detach().cpu().numpy().astype(np.int64)

        raise ValueError(self.ensemble_method)

    def fit_pred_features(
            self, 
            model,
            support_set_features: torch.Tensor, 
            support_label: torch.Tensor,
            query_set_features: torch.Tensor,
            query_label: torch.Tensor,
        ):

        A = support_set_features["attn_heads"].float() # (Ns, L, H, Dh)
        Ns, L, H, Dh = A.shape
        M = L * H
        A = A.reshape(Ns, M, Dh)

        Aq = query_set_features["attn_heads"].float() # (Nq, L, H, Dh)
        Nq, _, _, _ = Aq.shape
        Aq = Aq.reshape(Nq, L * H, Dh)


        # TRAINING PHASE
        support_heat_models_dict = self.get_rank_scores(A, support_label)

        labels_unique_tensor = support_heat_models_dict["labels_unique"]
        labels_unique = labels_unique_tensor.detach().cpu().numpy().astype(np.int64)
        head_scores = support_heat_models_dict["head_scores"]  # [L*H]
        order = torch.argsort(head_scores, descending=True)

        if self.context == "Class":
            selected_idx = order[: min(self.k, M)]
        elif self.context in {"Task", "Domain"}:
            selected_idx = torch.as_tensor(
                get_top_heads_V(dataset_name=self.current_dataset_name, mode=self.context),
                dtype=torch.long,
                device=order.device
            )[: min(self.k, M)]
        
        # INFERENCE PHASE
        query_predictions, query_logits = self.get_pred_labels(
            Aq[:, selected_idx, :],
            support_heat_models_dict["head_means"][selected_idx, :, :],
            support_heat_models_dict["head_cov_inv"][selected_idx, :, :],
            support_heat_models_dict["log_prior"],
        )  # [Nq, topk, K]
        

        yhat = self.ensemble(
            head_scores = head_scores[selected_idx],
            query_predictions = query_predictions,
            query_logits = query_logits,
            support_predictions = support_heat_models_dict["head_predictions"][:, selected_idx, :],
            support_logits = support_heat_models_dict["head_logits"][:, selected_idx, :],
            support_labels = support_label,
            )
        yq = query_label.detach().cpu().numpy().astype(np.int64)
        pred_prob = query_predictions.mean(dim=1).detach().cpu().numpy().astype(np.float32)  # [Nq, K]

        
        yhat = labels_unique[yhat]
        accuracy_final = float((yhat == yq).mean())

        return {
            "accuracy": accuracy_final,
            "pred_prob": pred_prob,
        }

    def fit_pred(
            self, 
            model, 
            support_set, 
            support_label, 
            query_set, 
            query_label
        ):
        
        # Inference
        support_set_features, query_set_features = self.get_frozen_features(model, support_set, query_set)

        # Classifier
        with torch.inference_mode():
            return self.fit_pred_features(
                model,
                support_set_features,
                support_label,
                query_set_features,
                query_label,
            )


class ClassifierHEC_T(Classifier):
    frozen_classifier = True
    
    def __init__(
        self,
        input_mode = None,
        transforms = None,
        log_losses: bool = False,
        k: int = 20,
        tau: float = 10.0,
        ensemble_method = "mean_prob",
        context = "Class", # "Class" or "Task" or "Domain"
        encoding = "lettre", # direct lettre
    ):
        super().__init__(
            input_mode=input_mode,
            transforms=transforms,
        )
        self.k = k
        self.tau = tau
        self.ensemble_method = ensemble_method
        self.encoding = encoding

        if context in {"Task", "Domain"} and log_losses:
            raise ValueError("Cannot log losses when context is Task or Domain, because there is only one head and no training is done.")

        self.context = context
        self.log_losses = log_losses
    
    def get_rank_scores(
            self, 
            support_features,      # [B, L*H, C]
            text_features, # [L*H, K, C]
            support_labels,        # [B]
        ):
        # Pre Prrocessing
        N = support_features.shape[0]
        M = support_features.shape[1]
        C = support_features.shape[2]
        K = text_features.shape[1]
        labels_unique = torch.unique(support_labels)
        y_idx = torch.searchsorted(labels_unique, support_labels)  # [B]

        # Normalize features
        support_features = F.normalize(support_features, p=2, dim=-1)                  # [B, L*H, C]
        text_features = F.normalize(text_features, p=2, dim=-1)                        # [L*H, K, C]
        
        text_features = text_features.to(support_features.dtype)
        l2_sq = (
            (support_features * support_features).sum(dim=-1)[:, :, None]
            - 2.0 * torch.einsum('bnc,nkc->bnk', support_features, text_features)
            + (text_features * text_features).sum(dim=-1)[None, :, :]
        )
        logits = -0.5 * l2_sq                                                          # [B, L*H, K]
        
        z = torch.softmax(logits / self.tau, dim=-1)                                   # [B, L*H, K]
        head_scores = z[torch.arange(support_labels.numel(), device=support_labels.device), :, y_idx].mean(dim=0)  # [L*H]
        
        return {
            "head_scores": head_scores,       # [L*H]
            "head_predictions": z,            # [B, L*H, K]
            "labels_unique": labels_unique,   # [K]
        }

    def get_pred_labels(
            self, 
            query_features,        # [Nq, topk, C]
            text_features, # [topk, K, C]
        ):
        query_features = F.normalize(query_features, p=2, dim=-1)                       # [Nq, topk, C]
        text_features = F.normalize(text_features, p=2, dim=-1)         # [topk, K, C]

        diff = query_features[:, :, None, :] - text_features[None, :, :, :]     # [Nq, topk, K, C]
        l2_sq = (diff * diff).sum(dim=-1)                                               # [Nq, topk, K]
        logits = -0.5 * l2_sq                                                           # [Nq, topk, K]

        z = torch.softmax(logits / self.tau, dim=-1)                                    # [Nq, topk, K]
        return z                                                                        # [Nq, topk, K]

    def get_heads_accuracy(
            self,
            query_features, # [Nq, topk, C]
            query_labels,   # [Nq]
            text_features,  # [topk, K, C]
            labels_unique,
        ):
        z = self.get_pred_labels(
            query_features,
            text_features,
        )  # [Nq, topk, K]

        y_hat = z.argmax(dim=-1)                                                       # [Nq, topk]
        y_idx = torch.searchsorted(labels_unique, query_labels)                        # [Nq]

        correct = (y_hat == y_idx[:, None]).to(z.dtype)                                # [Nq, topk]
        onehot = F.one_hot(y_idx, num_classes=labels_unique.numel()).to(z.dtype)       # [Nq, K]

        acc_num = torch.einsum('nt,nk->tk', correct, onehot)                           # [topk, K]
        acc_den = onehot.sum(dim=0)                                                    # [K]
        return acc_num / acc_den[None, :]                                              # [topk, K]

    def ensemble(
            self,
            head_scores,         # [M]
            query_predictions,   # [Nq, M, K]
            support_predictions, # [Ns, M, K] # ONLY BOR BAESIAN
            support_labels,      # [Ns]       # ONLY BOR BAESIAN
        ):

        if self.ensemble_method == "mean_prob":
            return query_predictions.mean(dim=1).argmax(dim=-1).detach().cpu().numpy().astype(np.int64)

        raise ValueError(self.ensemble_method)

    def fit_pred_features(
            self, 
            model,
            support_set_features: torch.Tensor, 
            support_label: torch.Tensor,
            query_set_features: torch.Tensor,
            query_label: torch.Tensor,
        ):

        support_attn_heads = support_set_features["attn_heads"].float() # [Ns, L, H, C]
        Ns, L, H, C = support_attn_heads.shape

        query_attn_heads = query_set_features["attn_heads"].float()     # [Nq, L, H, C]
        Nq, _, _, _ = query_attn_heads.shape


        if self.encoding == "direct":
            text_heads = model.backbone.forward_text_heads()["attn_heads"] # [K,L,H,C]
            K = text_heads.shape[0]
            # Permute to [L,H,K,C]
            text_heads = text_heads.permute(1, 2, 0, 3).contiguous() # [L,H,K,C]
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
        
        # Get in the M model Abstraction
        support_attn_heads = support_attn_heads.reshape(Ns, L * H, C)
        query_attn_heads = query_attn_heads.reshape(Nq, L * H, C)
        text_heads = text_heads.reshape(L * H, K, C)

        # TRAINING PHASE
        support_heat_models_dict = self.get_rank_scores(
            support_attn_heads, 
            text_heads,
            support_label,
        )

        labels_unique_tensor = support_heat_models_dict["labels_unique"]
        labels_unique = labels_unique_tensor.detach().cpu().numpy().astype(np.int64)
        head_scores = support_heat_models_dict["head_scores"]  # [L*H]
        order = torch.argsort(head_scores, descending=True)


        if self.context == "Class":
            selected_idx = order[: min(self.k, L*H)]
        elif self.context in {"Task", "Domain"}:
            selected_idx = torch.as_tensor(
                get_top_heads_T(dataset_name=self.current_dataset_name, mode=self.context, model_name=model.backbone.model_name, encoding=self.encoding),
                dtype=torch.long,
                device=order.device
            )[: min(self.k, L*H)]
        
        
        # INFERENCE PHASE
        query_predictions = self.get_pred_labels(
            query_attn_heads[:, selected_idx, :],
            text_heads[selected_idx, :, :],
        )  # [Nq, topk, K]
        

        yhat = self.ensemble(
            head_scores = head_scores[selected_idx],
            query_predictions = query_predictions,
            support_predictions = support_heat_models_dict["head_predictions"][:, selected_idx, :],
            support_labels = support_label,
            )
        yq = query_label.detach().cpu().numpy().astype(np.int64)
        pred_prob = query_predictions.mean(dim=1).detach().cpu().numpy().astype(np.float32)  # [Nq, K]
        
        yhat = labels_unique[yhat]
        accuracy_final = float((yhat == yq).mean())

        return {
            "accuracy": accuracy_final,
            "pred_prob": pred_prob,
        }

    def fit_pred(
            self, 
            model,
            support_set: torch.Tensor, 
            support_label: torch.Tensor,
            query_set: torch.Tensor,
            query_label: torch.Tensor,
        ):
        
        # Inference
        support_set_features, query_set_features = self.get_frozen_features(model, support_set, query_set)
        
        # Classification
        with torch.inference_mode():
            return self.fit_pred_features(
                model,
                support_set_features,
                support_label,
                query_set_features,
                query_label,
            )


class ClassifierHEC_VT(Classifier):
    frozen_classifier = True
    
    def __init__(
        self,
        input_mode = None,
        transforms = None,
        log_losses: bool = False,
        alpha: float = 1,
        k_v: int = 20,
        tau_v: float = 10.0,
        k_t: int = 7,
        tau_t: float = 10.0,
        ensemble_method = "mean_prob",
        context = "Class", # "Class" or "Task" or "Domain"
        encoding = "lettre",
    ):
        super().__init__(
            input_mode=input_mode,
            transforms=transforms,
        )
        if log_losses:
            raise NotImplementedError("Logging losses is not implemented for ClassifierHEC_VT")

        self.k_v = k_v
        self.tau_v = tau_v
        self.k_t = k_t
        self.tau_t = tau_t
        self.ensemble_method = ensemble_method
        self.alpha = alpha
        self.context = context


        self.hec_v = ClassifierHEC_V(
            input_mode=input_mode,
            transforms=transforms,
            log_losses=False,
            k=k_v,
            tau=tau_v,
            ensemble_method=ensemble_method,
            context = context,
        )

        self.hec_t = ClassifierHEC_T(
            input_mode=input_mode,
            transforms=transforms,
            log_losses=False,
            k=k_t,
            tau=tau_t,
            ensemble_method=ensemble_method,
            context = context,
            encoding = encoding,

        )

    
    def fit_pred(
            self, 
            model,
            support_set: torch.Tensor, 
            support_label: torch.Tensor,
            query_set: torch.Tensor,
            query_label: torch.Tensor,
        ):
        
        # Inference
        support_set_features, query_set_features = self.get_frozen_features(model, support_set, query_set)
        
        self.hec_v.current_label_to_str = self.current_label_to_str
        self.hec_v.current_dataset_name = self.current_dataset_name
        self.hec_t.current_label_to_str = self.current_label_to_str
        self.hec_t.current_dataset_name = self.current_dataset_name
        # Classification
        hec_v_results = self.hec_v.fit_pred_features(
            model,
            support_set_features,
            support_label,
            query_set_features,
            query_label,
        )

        hec_t_results = self.hec_t.fit_pred_features(
            model,
            support_set_features,
            support_label,
            query_set_features,
            query_label,
        )

        final_prob = hec_t_results["pred_prob"] + self.alpha * hec_v_results["pred_prob"]  # np.ndarray [Nq,K]
        final_pred = final_prob.argmax(axis=-1)

        labels_unique = torch.unique(support_label).detach().cpu().numpy().astype(np.int64)
        yhat = labels_unique[final_pred]
        yq = query_label.detach().cpu().numpy().astype(np.int64)
        accuracy_final = float((yhat == yq).mean())
        
        return {
            "accuracy": accuracy_final,
            "pred_prob": final_prob,
        }


# ------------------------------------------- #
# ----------------   VLMs   ----------------- #
# ------------------------------------------- #



class ClassifierVLM(Classifier):
    frozen_classifier = True

    def encode(self, model, device):
        
        if model.backbone.type_model == 'LMM':
            prompts_dict = self.current_label_to_str  # {idx: class_str}
            self.unique_labels = torch.tensor(sorted(prompts_dict.keys()), device=device)
            prompts = [prompts_dict[int(i)] for i in self.unique_labels.tolist()]  # list[str]

            with torch.no_grad():
                prompt_set_text  = model.backbone.get_class_txt(prompts)['x_norm_txttoken']  # [K, C]

        elif model.backbone.type_model == 'CLIP':
            prompts = get_prompt_clip(self.current_label_to_str, self.current_dataset_name, self.current_benchmark_name)
            self.unique_labels = torch.tensor(sorted(prompts.keys()), device=device)
            prompts = [prompts[int(i)] for i in self.unique_labels.tolist()]

            with torch.no_grad():
                prompt_set_text  = model.backbone.forward_text(prompts)['x_norm_txttoken']  # [K, C]
        else:
            raise NotImplementedError(f"VLM type_model {model.backbone.type_model} not implemented.")
    
        return prompt_set_text

    
class ClassifierVLMBaseline(ClassifierVLM):
    frozen_classifier = True
    
    def __init__(
            self, 
            input_mode='cls',
            transforms=None,
            log_losses: bool = False,
            ):
        self.input_mode = input_mode
        self.unique_labels = None

    def fit_pred(
            self, model, 
            support_set, support_label, query_set, query_label
        ):


        with torch.no_grad():
            # Vision
            support_set_vision = model.infer(support_set)['x_norm_clstoken'] # [Ns, C]
            query_set_vision   = model.infer(query_set)['x_norm_clstoken']   # [Nq, C]
            # Text
            prompt_set_text = self.encode(model, device=query_label.device)
        
            # Normalize
            prompt_set_text  = F.normalize(prompt_set_text, dim=1).float()
            query_set_vision = F.normalize(query_set_vision, dim=1).float()

            

        logits = 1 * (query_set_vision @ prompt_set_text.t())  # [Nq, K]
        probs = F.softmax(logits, dim=1)
        scores, top_idx = probs.max(dim=1)

        query_label_pred = self.unique_labels.to(top_idx.device)[top_idx]
        accuracy = (query_label_pred == query_label).float().mean().item()

        return {
            "query_label_pred": query_label_pred,
            "query_label_pred_score": scores,
            "logits": logits,
            "accuracy": accuracy,
        }


class ClassifierVLMGDA(ClassifierVLM):
    # Reimplemntation from https://github.com/mrflogs/iclr24
    frozen_classifier = True
    
    def __init__(
            self,
            input_mode="cls",
            alpha: float = 1.0,
            transforms=None,
            log_losses: bool = False,
        ):
        self.input_mode = input_mode
        self.alpha = alpha

    def fit_pred(
            self, model,
            support_set, support_label, query_set, query_label
        ):
        
        with torch.no_grad():
            support_set_vision = get_proto(model.infer(support_set), self.input_mode) #["x_norm_clstoken"]  # [Ns, C]
            query_set_vision   = get_proto(model.infer(query_set), self.input_mode)  #["x_norm_clstoken"]  # [Nq, C]
            prompt_set_text    = self.encode(model, device=query_label.device) # [K, C]

            support_set_vision = F.normalize(support_set_vision, dim=1).float()  # [Ns, C]
            query_set_vision   = F.normalize(query_set_vision, dim=1).float()    # [Nq, C]
            prompt_set_text    = F.normalize(prompt_set_text, dim=1).float()     # [K,  C]

        clip_logits = 100.0 * (query_set_vision @ prompt_set_text.t())  # [Nq, K]

        vecs = support_set_vision
        labels = support_label

        mus = torch.stack([vecs[labels == lab].mean(dim=0) for lab in self.unique_labels], dim=0)  # [K, C]
        center_vecs = torch.cat(
            [vecs[labels == lab] - mus[i].unsqueeze(0) for i, lab in enumerate(self.unique_labels)],
            dim=0
        )  # [N, C]

        N = center_vecs.shape[0]
        D = center_vecs.shape[1]
        cov = (center_vecs.t() @ center_vecs) / (N - 1)  # [C, C]
        cov_inv = D * torch.linalg.pinv((N - 1) * cov + cov.trace() * torch.eye(D, device=vecs.device, dtype=vecs.dtype))

        K = self.unique_labels.numel()
        ps = torch.ones(K, device=vecs.device, dtype=vecs.dtype) / K

        W = cov_inv @ mus.t()  # [C, K]
        b = ps.log() - 0.5 * torch.einsum('kc,cd,kd->k', mus, cov_inv, mus)  # [K]

        gda_logits = query_set_vision @ W + b  # [Nq, K]
        logits = clip_logits + self.alpha * gda_logits  # [Nq, K]

        probs = F.softmax(logits, dim=1)
        scores, top_idx = probs.max(dim=1)
        query_label_pred = self.unique_labels.to(top_idx.device)[top_idx]
        accuracy = (query_label_pred == query_label).float().mean().item()

        return {
            "query_label_pred": query_label_pred,
            "query_label_pred_score": scores,
            "logits": logits,
            "accuracy": accuracy,
        }
