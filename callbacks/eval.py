# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


class KGCNMetric(Callback):
    def __init__(self, x_train, y_train, x_valid, y_valid, n_item, user_num=100, threshold=0.5):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.n_item = n_item
        self.user_num = user_num
        self.threshold = threshold

        self.user_list, self.train_record, self.valid_record, \
            self.item_set, self.k_list = self.topk_settings()

        super(KGCNMetric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_valid).flatten()
        y_true = self.y_valid.flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)

        y_pred = [1 if prob >= self.threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        topk_p, topk_r = self.topk_eval()

        logs['val_auc'] = auc
        logs['val_acc'] = acc
        logs['val_f1'] = f1
        logs['val_topk_p'] = topk_p
        logs['val_topk_r'] = topk_r
        print(f'Logging Info - epoch: {epoch+1}, val_auc: {auc}, val_acc: {acc}, val_f1: {f1},'
              f'val_topk_p: {topk_p}, val_topk_r: {topk_r}')

    def topk_settings(self):
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = self.get_user_record(np.concatenate([self.x_train[0], self.x_train[1],
                                                            self.y_train], axis=-1), True)
        valid_record = self.get_user_record(np.concatenate([self.x_valid[0], self.x_valid[1],
                                                            self.y_valid], axis=-1), False)
        user_list = list(set(train_record.keys()) & set(valid_record.keys()))
        if len(user_list) > self.user_num:
            user_list = np.random.choice(user_list, size=self.user_num, replace=False)
        item_set = set(list(range(self.n_item)))
        return user_list, train_record, valid_record, item_set, k_list

    @staticmethod
    def get_user_record(data, is_train):
        user_history_dict = defaultdict(set)
        for interaction in data:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if is_train or label == 1:
                user_history_dict[user].add(item)
        return user_history_dict

    def topk_eval(self):
        precision_list = {k: [] for k in self.k_list}
        recall_list = {k: [] for k in self.k_list}

        for user in self.user_list:
            test_item_list = list(self.item_set - self.train_record[user])
            item_score_map = dict()

            input_user = np.expand_dims(np.array([user] * len(test_item_list)), axis=1)
            input_item = np.expand_dims(np.array(test_item_list), axis=1)
            item_scores = self.model.predict([input_user, input_item])
            for item, score in zip(test_item_list, item_scores):
                item_score_map[item] = score

            item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1],
                                            reverse=True)
            item_sorted = [i[0] for i in item_score_pair_sorted]

            for k in self.k_list:
                hit_num = len(set(item_sorted[:k]) & self.valid_record[user])
                precision_list[k].append(hit_num / k)
                recall_list[k].append(hit_num / len(self.valid_record[user]))

        precision = [np.mean(precision_list[k]) for k in self.k_list]
        recall = [np.mean(recall_list[k]) for k in self.k_list]
        return precision, recall
