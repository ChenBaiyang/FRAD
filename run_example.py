import numpy as np
import torch
from sklearn.preprocessing import minmax_scale

torch.set_default_dtype(torch.float32)
from models import FRAD_example

if __name__ == "__main__":
    data = np.array(
        [[2, 0.2, 1],
         [1, 0.1, 2],
         [9, 0.5, 2],
         [5, 0.8, 1],
         [7, 0.2, 2],
         [3, 0.7, 1]])

    # column one is a nominal attribute, which doesnot apply to min-max.
    data[:, :-1] = minmax_scale(data[:, :-1])
    print(data)

    out_factor = FRAD_example(data, dim=3, gamma=0.4)
    print(out_factor)
