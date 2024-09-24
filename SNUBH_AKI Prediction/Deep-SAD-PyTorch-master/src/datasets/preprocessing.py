import torch
import numpy as np


def create_semisupervised_setting(labels, sensitive_attr, normal_classes, outlier_classes, known_outlier_classes,
                                  ratio_known_normal, ratio_known_outlier, ratio_pollution, random_state):
    """
    Create a semi-supervised data setting.

    :param labels: np.array with labels of all dataset samples
    :param sensitive_attr: np.array with sensitive attribute values (e.g., 'gender')
    :param normal_classes: tuple with normal class labels
    :param outlier_classes: tuple with anomaly class labels
    :param known_outlier_classes: tuple with known (labeled) anomaly class labels
    :param ratio_known_normal: the desired ratio of known (labeled) normal samples
    :param ratio_known_outlier: the desired ratio of known (labeled) anomalous samples
    :param ratio_pollution: the desired pollution ratio of the unlabeled data with unknown (unlabeled) anomalies.
    :param random_state: seed for reproducibility
    :return: tuple with list of sample indices, list of original labels, list of semi-supervised labels, list of sensitive attributes
    """
    # random_state가 None이 아니면 정수로 변환 시도
    if random_state is not None:
        if isinstance(random_state, int):
            np.random.seed(random_state)
        else:
            try:
                random_state = int(random_state)
                np.random.seed(random_state)
            except ValueError:
                raise ValueError("random_state는 정수형 또는 None이어야 합니다.")

    # Indices for each class
    idx_normal = np.where(np.isin(labels, normal_classes))[0]
    idx_outlier = np.where(np.isin(labels, outlier_classes))[0]
    idx_known_outlier_candidates = np.where(np.isin(labels, known_outlier_classes))[0]

    n_normal = len(idx_normal)
    n_outlier = len(idx_outlier)
    n_known_outlier_candidates = len(idx_known_outlier_candidates)

    # Calculate number of known and unlabeled samples
    n_known_normal = int(n_normal * ratio_known_normal)
    n_unlabeled_normal = n_normal - n_known_normal

    n_known_outlier = int(n_known_outlier_candidates * ratio_known_outlier)
    n_unlabeled_outlier = n_outlier - n_known_outlier

    # Sample known and unlabeled normals
    perm_normal = np.random.permutation(idx_normal)
    idx_known_normal = perm_normal[:n_known_normal]
    idx_unlabeled_normal = perm_normal[n_known_normal:]

    # Sample known and unlabeled outliers
    if n_known_outlier > 0:
        perm_known_outlier = np.random.permutation(idx_known_outlier_candidates)
        idx_known_outlier = perm_known_outlier[:n_known_outlier]
    else:
        idx_known_outlier = np.array([], dtype=int)

    if n_unlabeled_outlier > 0:
        # Exclude known outliers from outlier indices
        idx_unlabeled_outlier = np.setdiff1d(idx_outlier, idx_known_outlier, assume_unique=True)
    else:
        idx_unlabeled_outlier = np.array([], dtype=int)

    # Assign semi_labels
    # Convention:
    # 0: known normal
    # 1: known outlier
    # -1: unlabeled normal or outlier
    semi_labels = np.full(n_normal + n_outlier, -1, dtype=np.int32)

    # Assign known normals
    semi_labels[:n_known_normal] = 0

    # Assign known outliers
    if n_known_outlier > 0:
        semi_labels[-n_known_outlier:] = 1

    # Apply pollution: flip some semi_labels in the unlabeled outlier data
    # Assuming pollution here means mislabeling some unlabeled outliers as known outliers
    if ratio_pollution > 0.0 and n_unlabeled_outlier > 0:
        n_polluted = int(n_unlabeled_outlier * ratio_pollution)
        if n_polluted > 0:
            polluted_indices = np.random.choice(n_unlabeled_outlier, n_polluted, replace=False)
            # Find the corresponding indices in the overall dataset
            polluted_global_indices = idx_unlabeled_outlier[polluted_indices]
            semi_labels[polluted_global_indices] = 1  # Assign as known outliers

    # Collect indices and labels
    list_idx = np.concatenate([idx_known_normal, idx_unlabeled_normal, idx_unlabeled_outlier, idx_known_outlier])
    list_labels = labels[list_idx].tolist()
    list_semi_labels = semi_labels[list_idx].tolist()
    sensitive_attr_list = sensitive_attr[list_idx].tolist()

    return list_idx.tolist(), list_labels, list_semi_labels, sensitive_attr_list
