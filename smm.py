import functools
import jax
import jax.numpy as jnp
import numpy as onp
import time 
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

#### TODO - Decide what to do here about the looping (need to be the commented
# version if we want to do distance matrix computations on 1000+ samples)
@functools.partial(jax.jit, static_argnums=(0))
def distmat(func: Callable, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """distance matrix"""
    #inner_part = lambda x_1: jax.vmap(lambda y_1: func(x_1, y_1))(y)
    #return jax.lax.map(inner_part, x)
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)

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

#### TODO - Needs checking 
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

#### TODO - Can we just normalise the data first?

params = {"gamma": 1}


start_time = time.time()

no_samples = 10000
no_sample_points = 100
x_train = onp.zeros((no_samples, no_sample_points), dtype=onp.float32)
y_train = onp.zeros((no_samples, ), dtype=onp.float32)

for i in range(no_samples):
    
    if i % 2 == 0:
        x_train[i, :] = onp.random.normal(1, 1, no_sample_points)
        y_train[i] = 1
    else:
        x_train[i, :] = onp.random.normal(10, 1, no_sample_points)
        y_train[i] = 2 
     

def compute_batch_svm(rbf_kernel, params, x_train_subset, y_train_subset):
    
    gram_mat = compute_gram(rbf_kernel, params, x_train_subset, x_train_subset, False)

    st = time.time()
    model = svm.SVC(kernel = 'precomputed')

    model.fit(gram_mat, y_train_subset)
    print(time.time() - st)
    support_vectors = x_train_subset[model.support_]
    support_vector_labels = y_train_subset[model.support_]
    
    return support_vectors, support_vector_labels

steps = int(no_samples/100)

support_vector_sets, support_vector_label_sets = zip(*Parallel(n_jobs=cpu_count())(delayed(compute_batch_svm)\
                  (rbf_kernel, params, x_train[i*100:(i+1)*100], y_train[i*100:(i+1)*100])\
                    for i in range(steps)))

support_vector_sets = list(support_vector_sets)
support_vector_label_sets = list(support_vector_label_sets)

x_train = support_vector_sets[0]
y_train = support_vector_label_sets[0]

support_vector_sets.pop()
support_vector_label_sets.pop()

for support_vector_set, support_vector_labels in zip(support_vector_sets, support_vector_label_sets):

    x_train = onp.vstack((x_train, support_vector_set))
    y_train = onp.hstack((y_train, support_vector_labels))
    
gram_mat = compute_gram(rbf_kernel, params, x_train, x_train, False)

model = svm.SVC(kernel = 'precomputed')

model.fit(gram_mat, y_train)

end_time = time.time()

print("Time taken: ", end_time - start_time)