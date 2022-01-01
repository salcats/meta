import functools
import jax
import jax.numpy as jnp
import numpy as onp
import time 
import cdt
from matplotlib import pyplot as plt
from typing import Callable, Dict 
from jax import random
import sklearn
from sklearn.metrics.pairwise import rbf_kernel as rbfk
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import GridSearchCV


@functools.partial(jax.jit, static_argnums=(0))
def distmat(func: Callable, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """distance matrix"""
    inner_part = lambda x_1: jax.vmap(lambda y_1: func(x_1, y_1))(y)
    return jax.lax.map(inner_part, x)
    #return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)

@jax.jit
def pdist_squareform(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """squared euclidean distance matrix
    Notes
    -----
    This is equivalent to the scipy commands
    >>> from scipy.spatial.distance import pdist, squareform
    >>> dists = squareform(pdist(X, metric='sqeuclidean')
    """
    return distmat(sqeuclidean_distance, x, y)

@jax.jit
def sqeuclidean_distance(x: jnp.array, y: jnp.array) -> float:
    return jnp.sum((x - y) ** 2)

@jax.jit
def rbf_kernel(
                params: Dict[str, float],
                x: jnp.ndarray,
                y: jnp.ndarray
    ) -> jnp.ndarray:
    return jnp.mean(jnp.exp(-params["gamma"] * pdist_squareform(x, y)))

@jax.jit
def rq_kernel(
                params: Dict[str, float],
                x: jnp.ndarray,
                y: jnp.ndarray
    ) -> jnp.ndarray:
    return jnp.mean(jnp.power(1 + (1/(params["alpha"]*params["gamma"])) * pdist_squareform(x, y), -params["alpha"]))

@functools.partial(jax.jit, static_argnums=(0, 4))
def compute_gram(
            func: Callable,
            params: Dict[str, float],
            x: jnp.ndarray,
            y: jnp.ndarray,
            normalise: bool
    ) -> jnp.ndarray:

    inner_product = lambda x_1: jax.vmap(lambda y_1: func(params, x_1, y_1))(y)
    
    gram_mat = jax.lax.map(inner_product, x)
      
    gram_diag_mat = jnp.diag(jax.lax.rsqrt(gram_mat.diagonal()))

    if normalise:
        gram_mat = jnp.matmul(gram_diag_mat, jnp.matmul(gram_mat, gram_diag_mat))
            
    return gram_mat


def compute_batch_gram(kernel, params, x, y):
    
    gram_matrix = compute_gram(kernel, params, x , y , False)

    return gram_matrix


def generate_indices(no_samples, step_size):

    index_pairs = [[i*step_size, (i+1)*step_size] for i in range(int(no_samples/step_size))]

    indices_list = []

    no_pairs = len(index_pairs)

    for index_1 in range(no_pairs):
        for index_2 in range(index_1, no_pairs):
            indices_list.append(index_pairs[index_1] + index_pairs[index_2])

    return indices_list


def unpack_batched_gram_matrices(batched_gram_matrices, no_samples, indices_list):
    
    gram_matrix = onp.zeros((no_samples, no_samples))

    for i in range(len(indices_list)):
        indices = indices_list[i]
        gram_matrix[indices[0]:indices[1], indices[2]:indices[3]] = batched_gram_matrices[i]
        gram_matrix[indices[2]:indices[3], indices[0]:indices[1]] = batched_gram_matrices[i].T

    return gram_matrix


def compute_training_gram(X_train, kernel, params, no_samples, step_size):
    
    indices_list = generate_indices(no_samples, step_size)
    
    batched_gram_matrices = Parallel(n_jobs=cpu_count())(delayed(compute_batch_gram)\
                    (kernel, params, X_train[indices[0]:indices[1]],
                    X_train[indices[2]:indices[3]]) for indices in indices_list)

    gram_matrix = unpack_batched_gram_matrices(batched_gram_matrices, no_samples, indices_list)

    return gram_matrix


def generate_data(data_generator, no_samples, no_sample_points, rescale):
    
    X_df, labels_df = data_generator.generate(
                                            no_samples,
                                            npoints=int(no_sample_points*2),
                                            rescale=rescale
    )

    X_df["joint"] = X_df['A'].apply(lambda x: x.tolist()) + X_df['B'].apply(lambda x: x.tolist())

    X = onp.array(X_df["joint"].values.tolist())

    labels = labels_df.to_numpy().reshape(no_samples, ).astype(onp.float32)
    X_df.drop("joint", inplace=True, axis=1)


    return X, X_df, labels


def generate_classifier_scores(classifier, data, labels):
    
    predictions = onp.sign(classifier.predict(data)).astype(int)
    scores = (predictions == labels).astype(int)
    scores[scores==0] = -1
    
    return scores


def visualise_embedded_data(gram_matrix, y_train):
    X_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(gram_matrix)

    class_1 = X_embedded[y_train==1]
    class_2 = X_embedded[y_train==-1]
    plt.scatter(class_1[:, 0], class_1[:, 1], c = 'r')
    plt.scatter(class_2[:, 0], class_2[:, 1], c = 'b')

    plt.show()


no_samples = 100
no_sample_points = 50
step_size = 100 

data_generator = cdt.data.CausalPairGenerator("sigmoid_add", noise_coeff=0.2)

classifiers = {
                "RECI": cdt.causality.pairwise.RECI(),
                "ANM" : cdt.causality.pairwise.ANM(),    
                "CDS" : cdt.causality.pairwise.CDS(),
}

"""
                "IGCI" : cdt.causality.pairwise.IGCI()        

"""
def fit_smm(gram_matrix_train, gram_matrix_test, y_train, y_test):

    model = svm.SVC(kernel='precomputed', class_weight="balanced",  C=1)
    model.fit(gram_matrix_train, y_train)
    preds = model.predict(gram_matrix_test)
    
    return 1 - accuracy_score(y_test, preds)
    
def objective(params):
    
    X_train = params["X_train"]
    X_test = params["X_test"]
    classifier = params["classifier"]
    labels_train = params["labels_train"]
    labels_test = params["labels_test"]
    X_train_df = params["X_train_df"]
    X_test_df = params["X_test_df"]
    gamma = params["gamma"]
    
    kernel_params = {"gamma": gamma}
    
    gram_matrix_train = compute_gram(rbf_kernel, kernel_params, X_train , X_train , True) 
    gram_matrix_test = compute_gram(rbf_kernel, kernel_params, X_test , X_train , False)
    
    y_train = generate_classifier_scores(classifier, X_train_df, labels_train)
    y_test = generate_classifier_scores(classifier, X_test_df, labels_test)

    loss = fit_smm(gram_matrix_train, gram_matrix_test, y_train, y_test)        
        
    return {"loss": loss, 'status': STATUS_OK }
   
   
# Set up gamma hyperparameter search 
rescale = False
X_train, X_train_df, labels_train = generate_data(data_generator, no_samples, no_sample_points, rescale)
X_test, X_test_df, labels_test = generate_data(data_generator, 50, no_sample_points, rescale)

params = {
            "X_train": X_train,
            "X_test": X_test,
            "labels_train": labels_train,
            "labels_test": labels_test,
            "X_train_df": X_train_df,
            "X_test_df": X_test_df,
}

def find_optimal_hps(params, classifier, classifier_name):
    
    params["classifier"] = classifier
    
    space ={
            "gamma": hp.uniform("gamma", 1, 1e4),
    }

    space = {**space, **params}
    
    trials = Trials()   

    results = fmin(objective, space, algo=tpe.suggest, max_evals=5, trials=trials)

    results["classifier_name"] = classifier_name
    results["classifier"] = classifier
    results["accuracy"] = 1 - onp.min(trials.losses())

    return results
    


batched_results = Parallel(n_jobs=cpu_count())(delayed(find_optimal_hps)\
                    (params, classifier, classifier_name) for classifier_name, classifier\
                    in classifiers.items())
   

def fit_smm_with_hps(params):
    
    X_train = params["X_train"]
    X_test = params["X_test"]
    classifier = params["classifier"]
    labels_train = params["labels_train"]
    X_train_df = params["X_train_df"]
    X_test_df = params["X_test_df"]
    X_test_df = params["X_test_df"]

    gamma = params["gamma"]
    classifier = params["classifier"]

    kernel_params = {"gamma": gamma}
    
    gram_matrix_train = compute_gram(rbf_kernel, kernel_params, X_train , X_train , False) 
    gram_matrix_test = compute_gram(rbf_kernel, kernel_params, X_test , X_train , False)
    
    y_train = generate_classifier_scores(classifier, X_train_df, labels_train)
    y_test = generate_classifier_scores(classifier, X_test_df, labels_test)

    model = svm.SVC(kernel='precomputed', class_weight="balanced",  C=1)
    
    model.fit(gram_matrix_train, y_train)
    
    decisions = model.decision_function(gram_matrix_test)
    predictions = onp.sign(classifier.predict(X_test_df)).astype(int)

    print(params["classifier_name"], onp.sum(y_test[y_test==1])/len(y_test))

    return decisions, predictions
    
    
    
X_test, X_test_df, labels_test = generate_data(data_generator, 50, no_sample_points, rescale)


params = {
            "X_train": X_train,
            "X_test": X_test,
            "labels_train": labels_train,
            "labels_test": labels_test,
            "X_train_df": X_train_df,
            "X_test_df": X_test_df,
}

causal_predictions = onp.zeros((50, ))
causal_predictions_w_acc = onp.zeros((50, ))

for i in range(len(classifiers)):
        
    params = {**params, **batched_results[i]}
    print(batched_results[i]["gamma"])

    decisions, predictions = fit_smm_with_hps(params)
    
    causal_predictions += decisions*predictions
    causal_predictions_w_acc += decisions*predictions*batched_results[i]["accuracy"]

print("------------------------------------------------") 
print("Weighted ensemble")
print("------------------------------------------------")
print(accuracy_score(labels_test, onp.sign(causal_predictions)))
print(accuracy_score(labels_test, onp.sign(causal_predictions_w_acc)))
print("------------------------------------------------")

causal_predictions = onp.zeros((50, len(classifiers)))
causal_predictions_w_acc = onp.zeros((50, len(classifiers)))

for i in range(len(classifiers)):
        
    params = {**params, **batched_results[i]}

    decisions, predictions = fit_smm_with_hps(params)
    
    causal_predictions[:, i] = decisions*predictions
    causal_predictions_w_acc[:, i] += decisions*predictions*batched_results[i]["accuracy"]


def return_max_entries(array):
    return array[range(len(array)), onp.argmax(onp.abs(array), axis=1)]

causal_predictions = return_max_entries(causal_predictions)
causal_predictions_w_o_acc = return_max_entries(causal_predictions_w_acc)

print("------------------------------------------------")
print("Maximum ensemble")
print("------------------------------------------------")
print(accuracy_score(labels_test, onp.sign(causal_predictions)))
print(accuracy_score(labels_test, onp.sign(causal_predictions_w_acc)))
print("------------------------------------------------")


#### Things to try
## Try without optimisation of the kernel and more samples. 
## Try weighting the accuracy differently like (1 - loss^2) 
## Try with optimisation of just one kernel for all of them
## Try grid search C and just optimise the kernel
## Try with more samples
## Ignore SMM if < % accuracy?