#!/usr/bin/env python3

# Standard Imports
import os
import gc
import random
import logging
import torch
import numpy as np
from tqdm import tqdm

# Hydra Imports
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Import project modules
from src.fewshot.engine import FinetuneWrapper
from src.fewshot.utils import instantiate_grid_search, tg_tf

# Clip Dataset Imports
from src.data_clip import build_dataset
from src.data_clip.utils import EpisodicNShotKWayDataset

# Configure logging / env variables
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TQDM_DISABLE"] = "1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Logging is set up.")

# Dynamic GPU selection
def setup_device(cfg):
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict
    assert torch.cuda.is_available()
    n_gpus = torch.cuda.device_count()
    job = HydraConfig.get().job
    job_num = job.num    # 0..(n_jobs-1)
    gpu_id = job_num % n_gpus                            # round-robin over GPUs

    torch.cuda.set_device(gpu_id)
    with open_dict(cfg):                                 # update config device
        cfg.device = f"cuda:{gpu_id}"

    logger.info(f"[Hydra job #{job_num}] using GPU {gpu_id} / {n_gpus-1}")
    logger.info(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"cfg.device={cfg.device}, current={torch.cuda.current_device()}, name={torch.cuda.get_device_name()}")
    dev = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(dev)
    logger.info(f"GPU{dev}: free={free/1e9:.2f}GB total={total/1e9:.2f}GB")
    return gpu_id, job_num

# Move any object to a device (torch.Tensor, dict, list, tuple)
def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x

@hydra.main(version_base="1.3.2", config_path="./config/eval", config_name="config_clip")
def main(cfg: DictConfig) -> None:

    # ------    GET PARAMS    ------ #
    tg_tf() # Avoid anoying Tensorflow warnings
    _, job_id = setup_device(cfg)
    seed = cfg.seed
    device = cfg.device

    # ------ DATA PARAMS INIT ------ #
    image_size = cfg.dataset.image_size
    datasets_used = cfg.dataset.datasets
    
    
    # ------    MODEL INIT    ------ #
    logger.info("Disabling Peft: The model is loaded once")
    backbone = instantiate(
        cfg.backbone, 
        tuning_config=cfg.peft_c, 
        device=device
    )
    head = instantiate(
        cfg.head, 
        num_features_in=cfg.backbone.num_features_out
    )
    model = FinetuneWrapper(
        backbone=backbone, 
        head=head, 
        device=device,
    )
    model.init_peft(config=cfg.peft_c)
    model.to(device)


    results = {}
    for dataset_name in datasets_used:

        # ---------------- setup ----------------
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        results[dataset_name] = []

        # ------ RANDOM SEEDING ------ #
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"


        # ----- DATASET PIPELINE ----- #
        if dataset_name in ['cu_birds', 'traffic_sign']:
            BENCHMARK = 'MD'
            cfg.dataset.path_base = cfg.dataset.path_base_MD
        else:
            BENCHMARK = 'CD'
            cfg.dataset.path_base = cfg.dataset.path_base_CD

        if BENCHMARK == 'CD':
            dataset = build_dataset({'subsample_classes': 'all'}, dataset_name, cfg.dataset.path_base, -1)
            dataset_train_val = dataset.train_x + dataset.val
            dataset_test      = dataset.test

            dataloader_train_val = EpisodicNShotKWayDataset(
                data_source=dataset_train_val,
                num_query = cfg.dataset.n_query,
                num_support = cfg.dataset.n_support,
                num_ways = cfg.dataset.n_ways,
                num_episodes=cfg.dataset.n_tuning_task,
                image_size=image_size,
                seed=seed,
            )

            dataloader_test = EpisodicNShotKWayDataset(
                data_source=dataset_test,
                num_query = cfg.dataset.n_query,
                num_support = cfg.dataset.n_support,
                num_ways = cfg.dataset.n_ways,
                num_episodes=cfg.dataset.n_task,
                image_size=image_size,
                seed=seed,
            )
        elif BENCHMARK == 'MD':
            logger.info(f"Using MetaDataset Loader for {dataset_name} with seed {seed}")
            from src.fewshot.utils_md_data import get_meta_dataset_loader
            dataloader_train_val = get_meta_dataset_loader(cfg, cfg.dataset.n_tuning_task, seed+1, dataset_name)
            dataloader_test = get_meta_dataset_loader(cfg, cfg.dataset.n_task, seed, dataset_name)
            logger.info(f"Using MetaDataset Loader for {dataset_name} with seed {seed}")
            pass
        else:
            raise ValueError(f"Unknown benchmark {BENCHMARK}")

        # --------  CLASSIFIER TUNING  --------- #
        iterator = iter(dataloader_train_val)
        tqdm_pbar = tqdm(range(cfg.dataset.n_tuning_task), desc=f"Tuning {dataset_name}")
        grid_search_cfgs = instantiate_grid_search(cfg.classifier)
        classifier_cfg_results = {idx_cfg: 0.0 for idx_cfg in range(len(grid_search_cfgs))}

        if len(grid_search_cfgs) == 0 or cfg.dataset.n_tuning_task == 0:
            classifier_cfg_best = OmegaConf.create(OmegaConf.to_container(cfg.classifier, resolve=False)) 
            if "sweep_cfg" in classifier_cfg_best:
                del classifier_cfg_best["sweep_cfg"]
            logger.info(f"[TUNING SKIP] Classifier CFG {classifier_cfg_best}")
        else:
            logger.info(f"[START TUNING] {dataset_name} over {len(grid_search_cfgs)} classifier configs")
            for iteration_number in tqdm_pbar:
                batch = next(iterator)
                logger.info(f"Batch Loaded")

                # ---------------- Iterate ---------------- #
                for classifier_cfg_idx, classifier_cfg_instance in enumerate(grid_search_cfgs):

                    # Classifier Init
                    classifier_transform = instantiate(
                        cfg.augment_c,
                        image_size=image_size,
                    )
                    classifier = instantiate(
                        classifier_cfg_instance,
                        log_losses=cfg.log_losses,
                        transforms=classifier_transform,
                    )
                    classifier.is_finetuned=False
                
                    # Data Init
                    support_set, support_label, query_set, query_label, label_to_str = batch
                    support_set, support_label = support_set.to(device), support_label.to(device)
                    query_set  , query_label   = query_set.to(device), query_label.to(device)
                    logger.info(f"Shapes - Support Set: {support_set.shape} Support Label: {support_label.shape} Query Set: {query_set.shape} Query Label: {query_label.shape}")


                    if hasattr(model.backbone, 'init_task_specific_data'):
                        model.backbone.init_task_specific_data(support_set, support_label, label_to_str, dataset_name, cfg.prompt, BENCHMARK)

                    # ----------       Train       ----------- #
                    classifier.init_task_specific_data(label_to_str, dataset_name, BENCHMARK)
                    prediction = classifier.fit_pred(
                        model,
                        support_set, 
                        support_label, 
                        query_set,
                        query_label,
                    )

                    # ----------       Results       ----------- #
                    classifier_cfg_results[classifier_cfg_idx] += prediction["accuracy"]

                    # --    Print Memory    -- #
                    if iteration_number == 0 and classifier_cfg_idx == 0 and dataset_name == datasets_used[0]:
                        dev = torch.cuda.current_device()
                        free, total = torch.cuda.mem_get_info(dev)
                        logger.info(
                            f"[after first task {dataset_name}] GPU{dev} mem: "
                            f"free={free/1e9:.2f}GB total={total/1e9:.2f}GB"
                        )
                        logger.info(
                            f"[after first task {dataset_name}] Support Set Size: {support_set.shape} Query Set Size: {query_set.shape}"
                        )
                    
                    # --      Clean Up      -- #
                    del prediction, support_set, query_set
                    gc.collect()
                    torch.cuda.empty_cache()
        
            classifier_cfg_best_idx = max(classifier_cfg_results, key=classifier_cfg_results.get)
            classifier_cfg_best = grid_search_cfgs[classifier_cfg_best_idx]
            logger.info(f"[TUNING DONE] Best Classifier CFG Index {classifier_cfg_best_idx} - {classifier_cfg_best}")


        # -------------  TESTING ------------- #
        iterator = iter(dataloader_test)
        tqdm_pbar = tqdm(range(cfg.dataset.n_task), desc=f"Testing {dataset_name}")
        for iteration_number in tqdm_pbar:
            batch = next(iterator) # check seed working
            result = {}

            # ---------------- Init ---------------- #
            # Classifier Init
            classifier_transform = instantiate(
                cfg.augment_c,
                image_size=image_size,
            )
            classifier = instantiate(
                classifier_cfg_best,
                log_losses=cfg.log_losses,
                transforms=classifier_transform,
            )
        
            # Data Init
            support_set, support_label, query_set, query_label, label_to_str = batch
            support_set, support_label = support_set.to(device), support_label.to(device)
            query_set  , query_label   = query_set.to(device), query_label.to(device)

            if hasattr(model.backbone, 'init_task_specific_data'):
                model.backbone.init_task_specific_data(
                    support_set, 
                    support_label, 
                    label_to_str, 
                    dataset_name, 
                    cfg.prompt, 
                    BENCHMARK,
                )


            # ----------       Train       ----------- #
            classifier.init_task_specific_data(label_to_str, dataset_name, BENCHMARK)
            prediction = classifier.fit_pred(
                model,
                support_set, 
                support_label, 
                query_set,
                query_label,
            )
            
            # ----------       Results       ----------- #
            top_1_accuracy = prediction["accuracy"]
            result['top-1-accuracy'] = top_1_accuracy
            result['tuning'] = classifier_cfg_results
            result.update(model.get_num_train_params())
            # ---------------------------------------- #
            
            # --    Print Results   -- #
            results[dataset_name].append(result)
            top_1_accuracy_avg = np.mean([res['top-1-accuracy'] for res in results[dataset_name]])
            logger.info(f"ID {job_id:02}-{iteration_number:04} - Accuracy: {100*top_1_accuracy_avg:.1f} | Last {100*top_1_accuracy:.2f}%")
            tqdm_pbar.set_postfix(
                last=f"{100*top_1_accuracy:.1f}%", 
                avg=f"{100*top_1_accuracy_avg:.1f}%"
            )

            # --    Print Memory    -- #
            if iteration_number == 0:# and dataset_name == datasets_used[0]:
                dev = torch.cuda.current_device()
                free, total = torch.cuda.mem_get_info(dev)
                logger.info(
                    f"[after first task {dataset_name}] GPU{dev} mem: "
                    f"free={free/1e9:.2f}GB total={total/1e9:.2f}GB"
                )
                logger.info(
                    f"[after first task {dataset_name}] Support Set Size: {support_set.shape} Query Set Size: {query_set.shape} Support mean,std: {torch.mean(support_set):.4f},{torch.std(support_set):.4f} Query mean,std: {torch.mean(query_set):.4f},{torch.std(query_set):.4f}"
                )
            
            # --      Clean Up      -- #
            del prediction, support_set, query_set
            gc.collect()
            torch.cuda.empty_cache()
        

        # -- Save and Print Dataset Results -- #
        mean_acc = np.mean([res['top-1-accuracy'] for res in results[dataset_name]])
        logger.info(f"[FINAL] {job_id:02}-{dataset_name}: Accuracy {mean_acc:.4f}")
        np.save(os.path.join(hydra_output_dir, 'metrics.npy'), results)

    # BEST OVERALL TUNING CONFIG
    if cfg.dataset.n_tuning_task > 0 and len(grid_search_cfgs) > 0:
        overall = np.zeros(len(grid_search_cfgs), dtype=np.float64)
        for dataset_name in datasets_used:
            tuning = results[dataset_name][0]["tuning"]
            overall += np.array([tuning[i] for i in range(len(grid_search_cfgs))], dtype=np.float64) / cfg.dataset.n_tuning_task
        overall /= len(datasets_used)
        best_idx = int(np.argmax(overall))
        logger.info(f"[OVERALL BEST] idx={best_idx} score={overall[best_idx]:.4f}")
        logger.info(OmegaConf.to_yaml(grid_search_cfgs[best_idx]))


    # --      Clean Up      -- #
    del backbone, head, model
    gc.collect()
    torch.cuda.empty_cache()

    # Save results npy
    logger.info(f"Results saved in {hydra_output_dir}")

    # Save results tb
    final_metrics = {}
    for dataset_name, task_results in results.items():
        mean_acc = np.mean([res['top-1-accuracy'] for res in task_results])
        final_metrics[f'hparam/{dataset_name}_accuracy'] = mean_acc
    
    return np.mean(list(final_metrics.values()))

if __name__ == "__main__":
    main()