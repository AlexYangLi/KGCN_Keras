# -*- coding: utf-8 -*-

import os

RAW_DATA_DIR = './raw_data'
PROCESSED_DATA_DIR = './data'
LOG_DIR = './log'
MODEL_SAVED_DIR = './ckpt'

KG_FILE = {'movie': os.path.join(RAW_DATA_DIR, 'movie', 'kg.txt'),
           'music': os.path.join(RAW_DATA_DIR, 'music', 'kg.txt')}
ITEM2ENTITY_FILE = {'movie': os.path.join(RAW_DATA_DIR, 'movie', 'item_index2entity_id.txt'),
                    'music': os.path.join(RAW_DATA_DIR, 'music', 'item_index2entity_id.txt')}
RATING_FILE = {'movie': os.path.join(RAW_DATA_DIR, 'movie', 'ratings.csv'),
               'music': os.path.join(RAW_DATA_DIR, 'music', 'user_artists.dat')}
SEPARATOR = {'movie': ',', 'music': '\t'}
THRESHOLD = {'movie': 4, 'music': 0}
NEIGHBOR_SIZE = {'movie': 4, 'music': 8}

USER_VOCAB_TEMPLATE = '{dataset}_user_vocab.pkl'
ITEM_VOCAB_TEMPLATE = '{dataset}_item_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'

PERFORMANCE_LOG = 'kgcn_performance.log'


class ModelConfig(object):
    def __init__(self):
        self.neighbor_sample_size = 4 # neighbor sampling size
        self.embed_dim = 32  # dimension of embedding
        self.n_depth = 2    # depth of receptive field
        self.l2_weight = 1e-7  # l2 regularizer weight
        self.lr = 2e-2  # learning rate
        self.batch_size = 65536
        self.aggregator_type = 'sum'
        self.n_epoch = 50
        self.optimizer = 'adam'

        self.user_vocab_size = None
        self.item_vocab_size = None
        self.entity_vocab_size = None
        self.relation_vocab_size = None
        self.adj_entity = None
        self.adj_relation = None

        self.exp_name = None
        self.model_name = None

        # checkpoint configuration
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_auc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_auc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1

        self.callbacks_to_add = None

        # config for learning rating scheduler and ensembler
        self.swa_start = 3
