# -*- coding: utf-8 -*-

import os
import numpy as np
from collections import defaultdict

from sklearn.model_selection import train_test_split

from config import PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, ITEM2ENTITY_FILE, KG_FILE, \
    RATING_FILE, USER_VOCAB_TEMPLATE, ITEM_VOCAB_TEMPLATE, ENTITY_VOCAB_TEMPLATE, \
    RELATION_VOCAB_TEMPLATE, SEPARATOR, THRESHOLD, TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, \
    TEST_DATA_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, ModelConfig, NEIGHBOR_SIZE
from utils import pickle_dump, format_filename


def read_item2entity_file(file_path: str, item_vocab: dict, entity_vocab: dict):
    print(f'Logging Info - Reading item2entity file: {file_path}' )
    assert len(item_vocab) == 0 and len(entity_vocab) == 0
    with open(file_path, encoding='utf8') as reader:
        for line in reader:
            item, entity = line.strip().split('\t')
            item_vocab[item] = len(item_vocab)
            entity_vocab[entity] = len(entity_vocab)


def read_rating_file(file_path: str, separator: str, threshold: int, user_vocab: dict,
                     item_vocab: dict):
    print(f'Logging Info - Reading rating file: {file_path}')

    assert len(user_vocab) == 0 and len(item_vocab) > 0
    user_pos_rating = defaultdict(set)
    user_neg_rating = defaultdict(set)
    with open(file_path, encoding='utf8') as reader:
        for idx, line in enumerate(reader):
            if idx == 0:
                continue

            user, item, rating = line.strip().split(separator)[:3]
            if item not in item_vocab:
                continue    # only consider items that has corresponding entities

            if float(rating) >= threshold:
                user_pos_rating[user].add(item_vocab[item])
            else:
                user_neg_rating[user].add(item_vocab[item])

    print('Logging Info - Converting rating file...')
    all_item_id_set = set(item_vocab.values())
    rating_data = []
    for user, pos_item_id_set in user_pos_rating.items():
        user_vocab[user] = len(user_vocab)
        user_id = user_vocab[user]

        for item_id in pos_item_id_set:
            rating_data.append([user_id, item_id, 1])

        unwatched_set = all_item_id_set - pos_item_id_set
        if user in user_neg_rating:
            unwatched_set -= user_neg_rating[user]

        for item_id in np.random.choice(list(unwatched_set), size=len(pos_item_id_set),
                                        replace=False):
            rating_data.append([user_id, item_id, 0])

    rating_matrix = np.array(rating_data)
    print(f'Logging Info - num of users: {len(user_vocab)}, num of items: {len(item_vocab)}')
    print(f'Logging Info - size of rating data: {rating_matrix.shape}')
    print(f'Logging Info - splitting rating data....')

    # train : dev : test = 6 : 2 : 2
    train_data, valid_data = train_test_split(rating_data, test_size=0.4)
    valid_data, test_data = train_test_split(valid_data, test_size=0.5)

    return train_data, valid_data, test_data


def read_kg(file_path: str, entity_vocab: dict, relation_vocab: dict, neighbor_sample_size: int):
    print(f'Logging Info - Reading kg file: {file_path}')

    kg = defaultdict(list)
    with open(file_path, encoding='utf8') as reader:
        for line in reader:
            head, relation, tail = line.strip().split('\t')

            if head not in entity_vocab:
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)

            # undirected graph
            kg[entity_vocab[head]].append((entity_vocab[tail], relation_vocab[relation]))
            kg[entity_vocab[tail]].append((entity_vocab[head], relation_vocab[relation]))
    print(f'Logging Info - num of entities: {len(entity_vocab)}, '
          f'num of relations: {len(relation_vocab)}')

    print('Logging Info - Constructing adjacency matrix...')
    n_entity = len(entity_vocab)
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)

    for entity_id in range(n_entity):
        all_neighbors = kg[entity_id]
        n_neighbor = len(all_neighbors)

        sample_indices = np.random.choice(
            n_neighbor,
            neighbor_sample_size,
            replace=False if n_neighbor >= neighbor_sample_size else True
        )

        adj_entity[entity_id] = np.array([all_neighbors[i][0] for i in sample_indices])
        adj_relation[entity_id] = np.array([all_neighbors[i][1] for i in sample_indices])

    return adj_entity, adj_relation


def process_data(dataset: str, neighbor_sample_size: int):
    user_vocab = {}
    item_vocab = {}
    entity_vocab = {}
    relation_vocab = {}

    read_item2entity_file(ITEM2ENTITY_FILE[dataset], item_vocab, entity_vocab)
    train_data, dev_data, test_data = read_rating_file(RATING_FILE[dataset], SEPARATOR[dataset],
                                                       THRESHOLD[dataset], user_vocab, item_vocab)
    adj_entity, adj_relation = read_kg(KG_FILE[dataset], entity_vocab, relation_vocab,
                                       neighbor_sample_size)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, USER_VOCAB_TEMPLATE, dataset=dataset),
                user_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ITEM_VOCAB_TEMPLATE, dataset=dataset),
                item_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset),
                entity_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, RELATION_VOCAB_TEMPLATE, dataset=dataset),
                relation_vocab)

    train_data_file = format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, dataset=dataset)
    np.save(train_data_file, train_data)
    print('Logging Info - Saved:', train_data_file)

    dev_data_file = format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, dataset=dataset)
    np.save(dev_data_file, dev_data)
    print('Logging Info - Saved:', dev_data_file)

    test_data_file = format_filename(PROCESSED_DATA_DIR, TEST_DATA_TEMPLATE, dataset=dataset)
    np.save(test_data_file, test_data)
    print('Logging Info - Saved:', test_data_file)

    adj_entity_file = format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset)
    np.save(adj_entity_file, adj_entity)
    print('Logging Info - Saved:', adj_entity_file)

    adj_relation_file = format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset)
    np.save(adj_relation_file, adj_relation)
    print('Logging Info - Saved:', adj_entity_file)


if __name__ == '__main__':
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)

    model_config = ModelConfig()
    process_data('movie', NEIGHBOR_SIZE['movie'])
    process_data('music', NEIGHBOR_SIZE['music'])
