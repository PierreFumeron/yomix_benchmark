from scipy.stats import rankdata
import numpy as np


# Yomix compute signature function from commit 0c7fce3fb39ba6cb2c6a9f71a234c2d1f421c205
def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    mean_diff = mu1 - mu2
    std_diff = sigma1 - sigma2
    wasserstein = np.sqrt(mean_diff**2 + std_diff**2)
    return wasserstein


def compute_signature(adata, means, stds, obs_indices_A, obs_indices_B=None):
    # STEP 1: sort features using Wasserstein distances
    a2 = adata[obs_indices_A, :].X
    mu2_array = a2.mean(axis=0)
    sigma2_array = np.copy(a2.std(axis=0))
    if obs_indices_B is None:
        mu = means
        sigma1_array = np.copy(stds.to_numpy())
        mu1_array = (
            (mu * adata.n_obs - mu2_array * len(obs_indices_A))
            / (adata.n_obs - len(obs_indices_A))
        ).to_numpy()
    else:
        a1 = adata.X[obs_indices_B, :]
        mu1_array = a1.mean(axis=0)
        sigma1_array = np.copy(a1.std(axis=0))
    sigma1_array[sigma1_array < 1e-8] = 1e-8
    sigma2_array[sigma2_array < 1e-8] = 1e-8
    dist_list = wasserstein_distance(mu1_array, sigma1_array, mu2_array, sigma2_array)

    sorted_features = np.argsort(dist_list)[::-1]

    if obs_indices_B is None:
        ref_array = np.arange(adata.n_obs)
        rest_indices = np.arange(adata.n_obs)[~np.isin(ref_array, obs_indices_A)]
    else:
        rest_indices = obs_indices_B

    samples_A = obs_indices_A
    samples_B = rest_indices

    # Keep only 100 features:
    selected_features = sorted_features[:100]

    def all_mcc(scores1, scores2):
        l1 = scores1.shape[1]
        l2 = scores2.shape[1]

        scores1.sort(axis=1)
        scores2.sort(axis=1)

        all_scores = np.hstack((scores1, scores2))

        ranks = rankdata(all_scores, method="min", axis=1).astype(int)

        all_scores.sort(axis=1)

        ranks1 = ranks[:, :l1]
        ranks2 = ranks[:, l1:]

        def matthews_c(a_, b_, c_, d_, l1_, l2_):

            max_value = np.maximum(l1_, l2_)
            tp = a_ / max_value
            fp = b_ / max_value
            fn = c_ / max_value
            tn = d_ / max_value

            denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

            mcc = (tp * tn - fp * fn) / np.sqrt(denominator)
            return mcc

        def searchsorted2d(a_, b_):
            m, n = a_.shape
            max_num = np.maximum(a_.max(), b_.max()) + 1
            r = max_num * np.arange(a_.shape[0])[:, None]
            p_ = np.searchsorted((a_ + r).ravel(), (b_ + r).ravel()).reshape(m, -1)
            return p_ - n * (np.arange(m)[:, None])

        maxis = np.maximum(np.max(ranks1, axis=1), np.max(ranks2, axis=1))
        rng = (
            np.repeat(np.arange(l1 + l2), scores1.shape[0]).reshape(l1 + l2, -1).T + 1
        ).clip(max=maxis[:, None])

        a = np.minimum(searchsorted2d(ranks1, rng)[:, 1:], l1)
        b = l1 - a
        c = np.minimum(searchsorted2d(ranks2, rng)[:, 1:], l2)
        d = l2 - c

        results = matthews_c(a, b, c, d, l1, l2)

        idx = l1 + l2 - 2 - np.abs(results[:, ::-1]).argmax(axis=1)

        first_axis_range = (np.arange(scores1.shape[0]),)
        mccscores = results[first_axis_range, idx]
        return mccscores

    sc1 = adata[samples_A, selected_features].copy().X.T
    sc2 = adata[samples_B, selected_features].copy().X.T
    mccs = all_mcc(sc1, sc2)
    new_selected_features = selected_features[np.argsort(np.abs(mccs.flatten()))[::-1]]
    mcc_dict = dict(map(lambda i, j: (i, j), selected_features, mccs.flatten()))
    mcc_dict_abs = dict(
        map(lambda i, j: (i, j), selected_features, np.abs(mccs).flatten())
    )
    # new_selected_features = new_selected_features[:20]
    up_or_down_d = {
        ft: ("-" if mcc_dict[ft] > 0.0 else "+") for ft in new_selected_features
    }
    return new_selected_features, mcc_dict_abs, up_or_down_d
