import numpy as np
from scipy import spatial 
import utilities
import time 
import jax
import jax.numpy as jnp
import functools
from typing import Callable, Dict 
from jax.experimental import loops

no_samples= 50
no_sample_points = 50

x = np.random.rand(no_sample_points, no_samples)
y = x



#@functools.partial(jax.jit, static_argnums=(0))
def compute_gram_v1(
                    func: Callable,
                    params: Dict[str, float],
                    x: jnp.ndarray,
                    y: jnp.ndarray,
    ) -> jnp.ndarray:

    no_samples = x.shape[0]
    no_sample_points = x.shape[1]
    
    gram_matrix = jnp.zeros((no_samples, no_samples)) 

    for i in range(no_samples):
        gram_matrix = gram_matrix.at[i, i:no_samples].set(jax.vmap(lambda y_1: func(no_sample_points, params, x[i,:], y_1))(y[i:,:]))

    with loops.Scope() as s:
    
        s.gram_matrix = jnp.zeros((no_samples, no_samples)) 
  
        for i in s.range(s.gram_matrix.shape[0]):
            s.gram_matrix = s.gram_matrix.at[i, i:no_samples].set(jax.vmap(lambda y_1: func(no_sample_points, params, x[i,:], y_1))(y[i:,:]))
            
    inner_product = lambda x_1: jax.vmap(lambda y_1: func(no_sample_points, params, x_1, y_1))(y)
    
    gram_mat = jax.lax.map(inner_product, x)
 
    quit()

    return gram_mat

#@functools.partial(jax.jit, static_argnums=(0))
def rbf_kernel_v1(
                no_sample_points: float,
                params: Dict[str, float],
                x: jnp.ndarray,
                y: jnp.ndarray
    ) -> jnp.ndarray:
    return jnp.mean(jnp.exp(-params["gamma"] * pdist_v1(no_sample_points, x, y)))

#@functools.partial(jax.jit, static_argnums=(0))
def pdist_v1(no_samples: float, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """squared euclidean distance matrix
    Notes
    -----
    This is equivalent to the scipy commands
    >>> from scipy.spatial.distance import pdist, squareform
    >>> dists = squareform(pdist(X, metric='sqeuclidean')
    """
    I = jnp.ones((no_samples, ))
    return jnp.outer(jnp.power(x, 2), I) + jnp.outer(I, jnp.power(y, 2)) - 2*jnp.outer(x, y) 

kernel_params = {"gamma": 1}


x = jnp.asarray(x)
y = jnp.asarray(y)

st = time.time()
G_1 = compute_gram_v1(rbf_kernel_v1, kernel_params, x, y)
print("jnp: ", time.time() - st)
