import numpy as np
import math

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pe = np.arange(seq_length*d_model, dtype=np.float64).reshape(seq_length, d_model)
    rows, cols = pe.shape
    for i in range(rows):
        for j in range(cols):
            encoded_i = j / 2 
            shared_arg = i / (10000**(2*encoded_i / d_model))
            if j % 2 == 0:
                pe[i][j] = math.sin(shared_arg)
            else:
                encoded_i = (j - 1) / 2
                shared_arg = i / (10000**(2*encoded_i / d_model))
                pe[i][j] = math.cos(shared_arg)
    return pe
    pass