# -*- coding: utf-8 -*-

import os
import gc
import time

import numpy as np
from collections import defaultdict
from keras import backend as K
from keras import optimizers

from utils import load_data, pickle_load, format_filename, write_log
from models import KGCN
from config import ModelConfig, PROCESSED_DATA_DIR, USER_VOCAB_TEMPLATE, ENTITY_VOCAB_TEMPLATE, \
    RELATION_VOCAB_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, LOG_DIR, PERFORMANCE_LOG, \
    ITEM_VOCAB_TEMPLATE


os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.Adam(learning_rate, clipnorm=5)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))


def train(dataset, neighbor_sample_size, embed_dim, n_depth, l2_weight, lr, optimizer_type,
          batch_size, aggregator_type, n_epoch, callbacks_to_add=None, overwrite=False):
    config = ModelConfig()
    config.neighbor_sample_size = neighbor_sample_size
    config.embed_dim = embed_dim
    config.n_depth = n_depth
    config.l2_weight = l2_weight
    config.lr = lr
    config.optimizer = get_optimizer(optimizer_type, lr)
    config.batch_size = batch_size
    config.aggregator_type = aggregator_type
    config.n_epoch = n_epoch
    config.callbacks_to_add = callbacks_to_add

    config.user_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                             USER_VOCAB_TEMPLATE,
                                                             dataset=dataset)))
    config.item_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                             ITEM_VOCAB_TEMPLATE,
                                                             dataset=dataset)))
    config.entity_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                               ENTITY_VOCAB_TEMPLATE,
                                                               dataset=dataset)))
    config.relation_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                                 RELATION_VOCAB_TEMPLATE,
                                                                 dataset=dataset)))
    config.adj_entity = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE,
                                                dataset=dataset))
    config.adj_relation = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE,
                                                  dataset=dataset))

    config.exp_name = f'kgcn_{dataset}_neigh_{neighbor_sample_size}_embed_{embed_dim}_depth_' \
                      f'{n_depth}_agg_{aggregator_type}_optimizer_{optimizer_type}_lr_{lr}_' \
                      f'batch_size_{batch_size}_epoch_{n_epoch}'
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str

    # logger to log output of training process
    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type,
                 'epoch': n_epoch, 'learning_rate': lr}
    print('Logging Info - Experiment: %s' % config.exp_name)
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    model = KGCN(config)
    train_data = load_data(dataset, 'train')
    valid_data = load_data(dataset, 'dev')
    test_data = load_data(dataset, 'test')

    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()
        model.fit(x_train=[train_data[:, :1], train_data[:, 1:2]], y_train=train_data[:, 2:3],
                  x_valid=[valid_data[:, :1], valid_data[:, 1:2]], y_valid=valid_data[:, 2:3])
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S",
                                                                 time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    print('Logging Info - Evaluate over valid data:')
    model.load_best_model()
    auc, acc, f1 = model.score(x=[valid_data[:, :1], valid_data[:, 1:2]], y=valid_data[:, 2:3])

    user_list, train_record, valid_record, item_set, k_list = topk_settings(train_data,
                                                                            valid_data,
                                                                            config.item_vocab_size)
    topk_p, topk_r = topk_eval(model, user_list, train_record, valid_record, item_set, k_list)
    train_log['dev_auc'] = auc
    train_log['dev_acc'] = acc
    train_log['dev_f1'] = f1
    train_log['dev_topk_p'] = topk_p
    train_log['dev_topk_r'] = topk_r

    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over valid data based on swa model:')
        auc, acc, f1 = model.score(x=[valid_data[:, :1], valid_data[:, 1:2]], y=valid_data[:, 2:3])
        topk_p, topk_r = topk_eval(model, user_list, train_record, dev_record, item_set, k_list)
        train_log['swa_dev_auc'] = auc
        train_log['swa_dev_acc'] = acc
        train_log['swa_dev_f1'] = f1
        train_log['swa_dev_topk_p'] = topk_p
        train_log['swa_dev_topk_r'] = topk_r

    print('Logging Info - Evaluate over test data:')
    model.load_best_model()
    auc, acc, f1 = model.score(x=[test_data[:, :1], test_data[:, 1:2]], y=test_data[:, 2:3])

    user_list, train_record, test_record, item_set, k_list = topk_settings(train_data,
                                                                           test_data,
                                                                           config.item_vocab_size)
    topk_p, topk_r = topk_eval(model, user_list, train_record, test_record, item_set, k_list)
    train_log['test_auc'] = auc
    train_log['test_acc'] = acc
    train_log['test_f1'] = f1
    train_log['test_topk_p'] = topk_p
    train_log['test_topk_r'] = topk_r

    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over test data based on swa model:')
        auc, acc, f1 = model.score(x=[test_data[:, :1], test_data[:, 1:2]], y=test_data[:, 2:3])
        topk_p, topk_r = topk_eval(model, user_list, train_record, test_record, item_set, k_list)
        train_log['swa_test_auc'] = auc
        train_log['swa_test_acc'] = acc
        train_log['swa_test_f1'] = f1
        train_log['swa_test_topk_p'] = topk_p
        train_log['swa_test_topk_r'] = topk_r

    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG), log=train_log, mode='a')
    del model
    gc.collect()
    K.clear_session()


def topk_settings(train_data, test_data, n_item, user_num=100):
    k_list = [1, 2, 5, 10, 20, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))
    return user_list, train_record, test_record, item_set, k_list


def get_user_record(data, is_train):
    user_history_dict = defaultdict(set)
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            user_history_dict[user].add(item)
    return user_history_dict


def topk_eval(model, user_list, train_record, test_record, item_set, k_list):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()

        input_user = np.expand_dims(np.array([user] * len(test_item_list)), axis=1)
        input_item = np.expand_dims(np.array(test_item_list), axis=1)
        item_scores = model.predict([input_user, input_item])
        for item, score in zip(test_item_list, item_scores):
            item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    return precision, recall


if __name__ == '__main__':
    train(dataset='movie',
          neighbor_sample_size=4,
          embed_dim=32,
          n_depth=2,
          l2_weight=1e-7,
          lr=2e-2,
          optimizer_type='adam',
          batch_size=65536,
          aggregator_type='sum',
          n_epoch=50,
          callbacks_to_add=['modelcheckpoint', 'earlystopping'])
    train(dataset='movie',
          neighbor_sample_size=4,
          embed_dim=32,
          n_depth=2,
          l2_weight=1e-7,
          lr=2e-2,
          optimizer_type='adam',
          batch_size=65536,
          aggregator_type='concat',
          n_epoch=50,
          callbacks_to_add=['modelcheckpoint', 'earlystopping'])
    train(dataset='movie',
          neighbor_sample_size=4,
          embed_dim=32,
          n_depth=2,
          l2_weight=1e-7,
          lr=2e-2,
          optimizer_type='adam',
          batch_size=65536,
          aggregator_type='neigh',
          n_epoch=50,
          callbacks_to_add=['modelcheckpoint', 'earlystopping'])

    train(dataset='movie',
          neighbor_sample_size=8,
          embed_dim=16,
          n_depth=1,
          l2_weight=1e-4,
          lr=5e-4,
          optimizer_type='adam',
          batch_size=128,
          aggregator_type='sum',
          n_epoch=50,
          callbacks_to_add=['modelcheckpoint', 'earlystopping'])
    train(dataset='movie',
          neighbor_sample_size=8,
          embed_dim=16,
          n_depth=1,
          l2_weight=1e-4,
          lr=5e-4,
          optimizer_type='adam',
          batch_size=128,
          aggregator_type='concat',
          n_epoch=50,
          callbacks_to_add=['modelcheckpoint', 'earlystopping'])
    train(dataset='movie',
          neighbor_sample_size=8,
          embed_dim=16,
          n_depth=1,
          l2_weight=1e-4,
          lr=5e-4,
          optimizer_type='adam',
          batch_size=128,
          aggregator_type='neigh',
          n_epoch=50,
          callbacks_to_add=['modelcheckpoint', 'earlystopping'])
