import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt


def smape(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label()
    return 'SMAPE', 100 * np.mean(np.abs(predt - y) / ((np.abs(y) + np.abs(predt)) / 2))


def sudo_smape(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        y = dtrain.get_label()
        res = predt - y
        return np.where(res > 0, 1 / ((res + 1) + 1e-6), 2 * res)

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        y = dtrain.get_label()
        res = predt - y
        return np.where(res > 0, -1 / ((res + 1) + 1e-6) ** 2, 2)

    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess


class Model:
    def __init__(self, train, test, feature_names, cv_results=None, answer=None):
        self.train = train
        self.test = test
        self.feature_names = feature_names
        self.cv_results = cv_results
        self.answer = answer

    def cross_validation(self, bd_no, params, num_boost_round, folds, early_stopping_rounds):
        data = self.train.loc[self.train['bd_no'] == bd_no, self.feature_names]
        label = self.train.loc[self.train['bd_no'] == bd_no, 'target']
        dtrain = xgb.DMatrix(data=data, label=label)

        cv = xgb.cv(params=params,
                    dtrain=dtrain,
                    num_boost_round=num_boost_round,
                    folds=folds,
                    obj=sudo_smape,
                    maximize=False,
                    early_stopping_rounds=early_stopping_rounds,
                    shuffle=False,
                    custom_metric=smape)

        return cv

    def draw_train_test_smape(self, train_smape, test_smape):
        fig, ax = plt.subplots()
        ax.plot(train_smape, label='train')
        ax.plot(test_smape, label='test')
        ax.legend()
        plt.show()

    def run_cv(self, params, num_boost_round, folds, early_stopping_rounds):
        # runtime: about 48 min.
        smape_means = []
        smape_stds = []
        best_num_boost_rounds = []

        for bd_no in tqdm(range(1, 101)):
            cv = self.cross_validation(bd_no, params, num_boost_round, folds, early_stopping_rounds)
            smape_mean = cv['test-SMAPE-mean'].min()
            smape_std = cv['test-SMAPE-std'].min()
            best_num_boost_round = cv['test-SMAPE-mean'].argmin()
            smape_means.append(smape_mean)
            smape_stds.append(smape_std)
            best_num_boost_rounds.append(best_num_boost_round)

        self.cv_results = pd.DataFrame({'smape_mean': smape_means, 'smape_std': smape_stds, 'best_num_boost_round': best_num_boost_rounds})
        return self.cv_results

    def draw_cv_scores(self):
        if self.cv_results is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(1, 101), self.cv_results['smape_mean'], width=0.2)
            ax.axhline(y=5, c='r', linestyle='--')
            ax.set_title('Mean: %f | Std: %f' % (self.cv_results['smape_mean'].mean(), self.cv_results['smape_std'].mean()))
        else:
            raise Exception('run_cv() should be runned first!')

    def predict(self, params, best_num_boost_rounds):
        answer = []
        for bd_no in tqdm(range(1, 101)):
            train_data = self.train.loc[self.train['bd_no'] == bd_no, self.feature_names]
            test_data = self.test.loc[self.test['bd_no'] == bd_no, self.feature_names]
            train_label = self.train.loc[self.train['bd_no'] == bd_no, 'target']
            dtrain = xgb.DMatrix(data=train_data, label=train_label)
            dtest = xgb.DMatrix(data=test_data)

            model = xgb.train(params=params,
                              dtrain=dtrain,
                              num_boost_round=best_num_boost_rounds[bd_no - 1],
                              obj=sudo_smape,
                              custom_metric=smape)
            model.save_model('./model/model%d.model' % bd_no)

            y_pred = model.predict(dtest)
            answer.extend(y_pred.tolist())
            self.answer = answer

        return answer

    def draw_prediction(self, bd_no):
        if self.answer is not None:
            fig, ax = plt.subplots(figsize=(20, 4))
            ax.plot('dt', 'target', data=self.train[self.train['bd_no'] == bd_no])
            ax.plot(self.test.loc[self.test['bd_no'] == bd_no, 'dt'], self.answer[(168 * bd_no - 168):(168 * bd_no)])
            plt.show()
        else:
            raise Exception('predict() should be runned first!')
