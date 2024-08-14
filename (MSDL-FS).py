import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import euclidean_distances
import multiprocessing as mp

def construct_kernels(X, kernel_scale_factor=1):
    K_dis = euclidean_distances(np.transpose(X))
    epsilon = kernel_scale_factor * np.median(K_dis[~np.eye(K_dis.shape[0], dtype=bool)])
    K = np.exp(-(K_dis ** 2) / (2 * epsilon ** 2))
    return K

def learn_dictionary(samples, n_components, transform_alpha=1.0, fit_algorithm='cd', n_runs=1,dict_init='random' ):
    all_dicos = []
    errors = []

    for _ in range(n_runs):
        try:
            dico = DictionaryLearning(n_components=n_components, transform_alpha=transform_alpha,
                                      fit_algorithm=fit_algorithm, max_iter=1000, tol=1e-8,dict_init = dict_init,n_jobs=10,random_state=42)
            dico.fit(samples)
            reconstructed_samples = dico.transform(samples).dot(dico.components_)
            error = np.sum((samples - reconstructed_samples) ** 2)
            all_dicos.append(dico.components_)
            errors.append(error)

        except Exception as e:
            print("Failed to learn dictionary:", e)
            continue

    if all_dicos:
        sorted_indices = np.argsort(errors)
        middle_index = sorted_indices[len(sorted_indices) // 2]
        best_dico = all_dicos[middle_index]
    else:
        best_dico = None

    return best_dico

def encode_sample(x, dictionary):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=20,tol=1e-4,)
    omp.fit(dictionary.T, x.T)
    return omp.coef_

def reconstruct_feature(encoded_sample, dictionary):
    reconstructed_sample = encoded_sample @ dictionary
    return reconstructed_sample

def compute_feature_importance(X_train,y_train ,K,D_combined, D_feature, sample_baseline_error, feature_baseline_error,label_baseline_error ,gamma, i,X):
    X_masked = X_train.copy()
    X_masked[:, i] = np.mean(X_train[:, i])

    encoded_masked = np.array([encode_sample(x, D_combined) for x in X_masked])
    X_reconstructed_masked = np.array([reconstruct_feature(enc, D_combined) for enc in encoded_masked])

    sample_error_increase = np.sum((X_masked - X_reconstructed_masked) ** 2, axis=1)

    K_masked = construct_kernels(X_masked, kernel_scale_factor=gamma)

    encoded_features_masked = np.array([encode_sample(k, D_feature) for k in K_masked.T])
    K_reconstructed_masked = np.array([reconstruct_feature(enc, D_feature) for enc in encoded_features_masked]).T

    feature_error_increase = np.sum((K_masked - K_reconstructed_masked) ** 2)

    # label_error_masked = np.sum((encoded_masked @ X - y_train) ** 2, axis=1)
    # label_error_increase = np.sum(label_error_masked) - label_baseline_error
    y_pred = np.dot(encoded_masked, np.linalg.pinv(encoded_masked.T @ encoded_masked) @ encoded_masked.T @ y_train)
    label_error_increase = np.sum((y_train - y_pred) ** 2)

    print(f"  feature: {i}")

    return sample_error_increase, feature_error_increase,label_error_increase

def feature_selection(X_train, y_train,n_sample_components, n_feature_components, gamma,a,b,c):
    K = construct_kernels(X_train)
    n_clusters=10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_train)
    kmeans.fit(X_train)
    cluster_centers = kmeans.cluster_centers_

    dict_init = cluster_centers

    D_feature = learn_dictionary(K, n_components=20,dict_init = dict_init)
    D_sample_local = []
    for i in range(n_clusters):
        cluster_samples = X_train[labels == i]
        n_samples_in_cluster = len(cluster_samples)

        n_components_local = 1

        D_local = learn_dictionary(cluster_samples, n_components_local,dict_init = dict_init)
        D_sample_local.append(D_local)

    D_sample_local = np.concatenate(D_sample_local, axis=0,)

    D_sample_global = learn_dictionary(X_train, n_components=10,dict_init = dict_init)

    D_combined = np.concatenate((D_sample_global, D_sample_local), axis=0)

    D_sample = (D_combined)
    encoded_samples = encode_sample(X_train, D_sample)
    X_reconstructed_samples = np.array([reconstruct_feature(enc, D_sample) for enc in encoded_samples])

    C = encoded_samples
    D = y_train

    X = np.linalg.pinv(C.T @ C) @ C.T @ D

    encoded_features = encode_sample(K, D_feature)
    K_reconstructed = np.array([reconstruct_feature(enc, D_feature) for enc in encoded_features]).T

    sample_error =  np.sum((X_train - X_reconstructed_samples) ** 2, axis=1)
    sample_baseline_error = np.sum(sample_error)
    print(sample_baseline_error)

    feature_error = np.sum((K - K_reconstructed) ** 2)
    feature_baseline_error = feature_error

    label_error = np.sum((encoded_samples @ X - y_train) ** 2, axis=1)
    label_baseline_error = np.sum(label_error)

    n_features = X_train.shape[1]
    pool = mp.Pool(processes=20)
    results = [pool.apply_async(compute_feature_importance, args=(X_train,y_train ,K,D_sample, D_feature, sample_baseline_error, feature_baseline_error, label_baseline_error,gamma, i,X)) for i in range(n_features)]
    feature_errors = [r.get() for r in results]
    pool.close()
    pool.join()

    sample_errors, feature_errors,label_errors = zip(*feature_errors)

    sample_errors_normalized = (sample_errors - np.min(sample_errors)) / (np.max(sample_errors) - np.min(sample_errors))
    feature_errors_normalized = (feature_errors - np.min(feature_errors)) / (np.max(feature_errors) - np.min(feature_errors))
    label_error_normalized = (label_errors - np.min(label_errors)) / (np.max(label_errors) - np.min(label_errors))



    total_errors = a*sample_errors_normalized + b*feature_errors_normalized +  c*label_error_normalized

    ranked_features = sorted(zip(range(n_features), total_errors), key=lambda x: x[1], reverse=True)

    return [feature for feature, importance in ranked_features]