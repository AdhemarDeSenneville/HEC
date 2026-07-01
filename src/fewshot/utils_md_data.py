
import os
import torch
import numpy as np

from .utils import tg_tf
tg_tf()
import tensorflow as tf

import gin
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
from meta_dataset.data import config
from meta_dataset.data import sampling


to_torch_labels = lambda a: torch.from_numpy(a.numpy()).long()
to_torch_imgs = lambda a: torch.from_numpy(np.transpose((1+a.numpy())/2, (0, 3, 1, 2)))

ALL_DATASETS = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012', 'omniglot', 'quickdraw', 'vgg_flower', 'traffic_sign', 'mscoco']
USE_BILEVEL_ONTOLOGY_LIST = [False]*len(ALL_DATASETS)
USE_DAG_ONTOLOGY_LIST = [False]*len(ALL_DATASETS)

def get_meta_dataset_loader(cfg, n_task, seed, dataset_name):

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(seed)
    sampling.RNG.seed(seed)

    gin.clear_config()
    gin.parse_config_file(cfg.dataset.path_config)

    num_query = cfg.dataset.n_query if cfg.dataset.n_query != 'None' else None
    num_support = cfg.dataset.n_support if cfg.dataset.n_support != 'None' else None
    num_ways = cfg.dataset.n_ways if cfg.dataset.n_ways != 'None' else None
    variable_ways_shots = config.EpisodeDescriptionConfig(
        num_query=num_query, 
        num_support=num_support, 
        num_ways=num_ways,
    )

    dataset_records_path = os.path.join(cfg.dataset.path_base, dataset_name)
    dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
    

    # Fancy Special Cases
    use_dag_ontology = False
    use_bilevel_ontology = False
    if num_ways is None:
        if dataset_name == 'ilsvrc_2012':
            use_dag_ontology = True
        if dataset_name == 'omniglot':
            use_bilevel_ontology = True

    
    # Split Handeling
    # These two don't have a train/val/test split.
    #if dataset_name in ['traffic_sign', 'mscoco']:
    #    if split in ['val', 'test']:
    #        split = 'test'
    #        tf.random.set_seed(seed+1) 
    #        sampling.RNG.seed(seed+1)
    ## Gin stuff
    #if split == 'val':
    #    split = learning_spec.Split.VALID
    #elif split == 'test':
    #    split = learning_spec.Split.TEST
    #elif split == 'train':
    #    split = learning_spec.Split.TRAIN
    #else:
    #    raise ValueError(f"Unknown split {split}")
    
    if dataset_name in ['omniglot']:
        superclasses_per_split = dict(dataset_spec.superclasses_per_split)
        superclasses_per_split[learning_spec.Split.TRAIN] = (
            superclasses_per_split[learning_spec.Split.TRAIN]
            + superclasses_per_split[learning_spec.Split.VALID]
        )
        superclasses_per_split[learning_spec.Split.VALID] = 0
        dataset_spec = dataset_spec._replace(superclasses_per_split=superclasses_per_split)
    else:
        classes_per_split = dict(dataset_spec.classes_per_split)
        total = (classes_per_split[learning_spec.Split.TRAIN] +
                classes_per_split[learning_spec.Split.VALID] +
                classes_per_split[learning_spec.Split.TEST])
        classes_per_split[learning_spec.Split.TRAIN] = total
        classes_per_split[learning_spec.Split.VALID] = 0
        classes_per_split[learning_spec.Split.TEST] = 0
        dataset_spec = dataset_spec._replace(classes_per_split=classes_per_split)

    dataset_episodic = pipeline.make_one_source_episode_pipeline(
        dataset_spec=dataset_spec,
        use_dag_ontology=use_dag_ontology,
        use_bilevel_ontology=use_bilevel_ontology,
        episode_descr_config=variable_ways_shots,
        split=learning_spec.Split.TRAIN,
        image_size=cfg.dataset.image_size,
        shuffle_buffer_size=cfg.dataset.shuffle_buffer_size,
    )
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    dataset_episodic = dataset_episodic.with_options(opts)

    def data_loader():
        for i, (e, _) in enumerate(dataset_episodic):
            if i == n_task:
                break
            
            support_y = e[1].numpy().tolist()
            support_cid = e[2].numpy().tolist()
            class_names = dataset_spec.class_names
            
            idx2str = {}
            for y, cid in zip(support_y, support_cid):
                if y not in idx2str:
                    idx2str[y] = class_names[cid]   # y in {0..4}

            
            yield (
                to_torch_imgs(e[0]), 
                to_torch_labels(e[1]),
                to_torch_imgs(e[3]), 
                to_torch_labels(e[4]),
                idx2str
            )

            
    return data_loader()

