from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics


def _init_parser():
    parser = ArgumentParser()
    parser.add_argument('--preds_paths', type=str, nargs='+', default=('out_test/preds_effv2.csv',))
    return parser


def main(args):
    ids = None
    labels = None
    preds = None
    for path in args.preds_paths:
        df = pd.read_csv(path)
        if ids is None:
            ids = list(df['id'].values)
            labels = df['label'].values
            preds = [df.iloc[:, 2:4].values]
        else:
            assert all([ids[i] == df['id'].iloc[i] for i in range(len(ids))])
            assert all([labels[i] == df['label'].iloc[i] for i in range(len(ids))])
            preds.append(df.iloc[:, 2:4].values)
    preds = np.stack(preds).mean(axis=0)
    preds_lbl = np.argmax(preds, axis=1)
    acc = metrics.accuracy_score(labels, preds_lbl)
    auroc = metrics.roc_auc_score(labels, preds[:, 1])
    print(f'acc: {acc:.03f}')
    print(f'auroc: {auroc:.03f}')


if __name__ == '__main__':
    parser = _init_parser()
    args = parser.parse_args()
    main(args)
