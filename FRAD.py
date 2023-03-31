import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from models import FRAD_torch
torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    fnames = 'data/SGCC_{}.npy'
    for fname in ['1000','5000','10000']:
        f = fnames.format(fname)
        dataset = np.load(f)
        data, label = dataset[:, :-1], dataset[:, -1]
        print("Total samples:{}  Outlier counts:{}".format(len(label), label.sum()))
        print("Data shape:{}".format(data.shape))

        deltas = [2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1]
        for delta in deltas:
            dim = data.shape[1]
            out_factor = FRAD_torch(data, dim=dim, gamma=delta)
            auc = roc_auc_score(label, out_factor)
            pr = average_precision_score(y_true=label, y_score=out_factor, pos_label=1)

            print('{}\tdelta={:.5f}\tmodel={}\tAUC={:.4f}\tPR={:.4f}'.format(f, delta, 'FRAD', auc, pr))
            record = str([f, delta, 'FRAD', auc, pr]) + '\n'
            open('results_FRAD.txt', 'a').write(record)



