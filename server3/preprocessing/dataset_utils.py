import numpy as np

def pad_or_trim(seq, T=60):
    if len(seq) > T:
        return seq[:T]
    if len(seq) < T:
        pad = np.zeros((T - len(seq), 42, 3))
        return np.concatenate([seq, pad], axis=0)
    return seq


def normalize_sequence(seq):
    """
    wrist-centered normalization per frame
    """
    seq = np.array(seq)

    # avoid divide issues
    wrist = seq[:, 0:1, :]
    seq = seq - wrist

    return seq
