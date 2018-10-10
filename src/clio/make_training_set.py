import random
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_pairs(x, indices, num_classes):
    """Positive and negative pair creation. Alternates between positive and
    negative pairs.

    Args:
        x (): Encoded documents. Can be a BoW or TF-IDF matrix.
        indices ():
        num_classes (int): Number of classes to consider. In this case, it is
                            MAK topics.

    Returns:
        pairs (list, str): Document pairs.
        labels (list, int): Label pairs. Takes binary values.

    """
    pairs = []
    labels = []

    # balance classes by setting the number of examples for each class equal to the size of the smallest example
    n = min([len(indices[d]) for d in range(num_classes)]) - 1

    for d in range(num_classes):
        for i in range(n):
            # positive
            z1, z2 = indices[d][i], indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]

            # negative
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = indices[d][i], indices[dn][i]
            pairs += [[x[z1], x[z2]]]

            # add labels
            labels += [1, 0]

    return pairs, labels

def split_pairs(pairs, i):
    arr = np.array([pair[i] for pair in pairs])
    return arr

def label_encoding(values):
    """Encode labels.

    Args:
        values (list, str)

    Returns:
        y_encoded (array): Encoded values.

    """
    y = np.array(values)
    # Instantiate encoder
    l = LabelEncoder()
    y_encoded = l.fit_transform(y)
    return y_encoded
