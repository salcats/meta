import functools
import jax
import jax.numpy as jnp
import numpy as onp
import cdt
from typing import Callable, Dict 
from joblib import Parallel, delayed

#### TODO - Try changing to 3d array 
def pdist_squareform(xy_i: jnp.ndarray, xy_j: jnp.ndarray, no_sample_points: int) -> jnp.ndarray:
    """  Computes the squared euclidean distance matrix.

    This function computes the squared euclidean distance matrix between two empircal joint
    distributions P_(X_i, Y_i), Q_(X_j, Y_j) where
    
    -- (x^i_k, y^i_k) are samples drawn from the joint P_(X_i, Y_i) 
    -- (x^j_k, y^j_k) are samples drawn from the joint P_(X_j, Y_j) 


    Parameters
    -----------
       

    Returns
    -----------

    """
    I = jax.device_put(jnp.ones((no_sample_points, )))
    
    x_i  = jax.lax.dynamic_slice(xy_i, (0,), (no_sample_points,))
    y_i  = jax.lax.dynamic_slice(xy_i, (no_sample_points,), (no_sample_points,))

    x_j  = jax.lax.dynamic_slice(xy_j, (0,), (no_sample_points,))
    y_j  = jax.lax.dynamic_slice(xy_j, (no_sample_points,), (no_sample_points,))
    
    return jnp.outer(jnp.power(x_i, 2) + jnp.power(y_i, 2), I)\
         + jnp.outer(I, jnp.power(x_j, 2) + jnp.power(y_j, 2))\
         - 2*(jnp.outer(x_i, x_j) + jnp.outer(y_i, y_j))  

def rbf_kernel( 
                x: jnp.ndarray,
                y: jnp.ndarray,    
                params: Dict[str, float],       
                no_sample_points: int,
    ) -> jnp.ndarray:
    """   Computes the inner product between two samples x, y constructed from
    RBF kernel.
    
    This function computes the inner product between two samples x, y 
    from the embedded probability distributions mu(P)_k and mu(Q)_k. The
    kernel of the RKHS is the RBF kernel. 
    
    Parameters
    -----------
       

    Returns
    -----------

    """
    return jnp.mean(jnp.exp(-params["gamma"] * pdist_squareform(x, y, no_sample_points)))

def rq_kernel(
                x: jnp.ndarray,
                y: jnp.ndarray,
                no_sample_points: float,
                params: Dict[str, float],
    ) -> jnp.ndarray:
    """  Computes the inner product between two samples x, y in a RKHS constructed from 
    RQ kernel.
    
    This function computes the inner product between two samples x, y 
    from the embedded probability distributions mu(P)_k and mu(Q)_k. The
    kernel of the RKHS is the RQ kernel. 
    
    Parameters
    -----------
       

    Returns
    -----------

    """ 
    return jnp.mean(jnp.power(1 + (params["factor"]*pdist_squareform(x, y, no_sample_points)), -params["alpha"]))

@functools.partial(jax.jit, static_argnums=(0, 5))
def compute_gram(
                    func: Callable,
                    params: Dict[str, float],
                    x: jnp.ndarray,
                    y: jnp.ndarray,
    ) -> jnp.ndarray:
    """ Computes the gram matrix between n samples x, y. 
    
    This function computes the gram matrix between n samples x, y from 
    the embedded probability distributions mu(P)_k and mu(Q)_k.
    
    Parameters
    -----------
       

    Returns
    -----------

    """    
    # Determine the number of points in each sample.
    no_sample_points = int(x.shape[1]/2)
    
    # Define the inner loop i.e. k(x_i, y_j) for all y_j.
    inner_loop = lambda x_1: jax.vmap(lambda y_1: func(x_1, y_1, params, no_sample_points))(y)
    
    # Compute the outer loop, i.e. the inner loop for all x_i.
    gram = jax.lax.map(inner_loop, x)
                 
    return gram


@functools.partial(jax.jit, static_argnums=(0, 4))
def compute_gram_gpu(
                    func: Callable,
                    params: Dict[str, float],
                    x: jnp.ndarray,
                    y: jnp.ndarray,
    ) -> jnp.ndarray:
    """ Computes the gram matrix between n samples x, y. 
    
    This function computes the gram matrix between n samples x, y from 
    the embedded probability distributions mu(P)_k and mu(Q)_k.
    
    Parameters
    -----------
       

    Returns
    -----------

    """    
    # Move data to GPU
    x = jax.device_put(x)
    y = jax.device_put(y)

    # Determine the number of points in each sample.
    no_sample_points = x.shape[1]
        
    # Compute the outer loop, i.e. the inner loop for all x_i.
    gram = jax.vmap(lambda x_1, y_1: func(x_1, y_1, params, no_sample_points))(y)(x)

    #### If that doesn't work then try this
    """
    # Define the inner loop i.e. k(x_i, y_j) for all y_j.
    inner_loop = lambda x_1: jax.vmap(lambda y_1: func(x_1, y_1, params, no_sample_points))(y)
    
    # Compute the outer loop, i.e. the inner loop for all x_i.
    gram = jax.lax.map(inner_loop, x)

    """

    return gram

#### TODO - Add some routine for making the pairs the same number of sample points.
def generate_data(
                    data_generator: cdt.data.causal_pair_generator.CausalPairGenerator,
                    no_samples: int,
                    no_sample_points: int,
                    rescale: bool
    ) -> list:
    """ Generates data. 
    
    This function generates data ... 
    
    Parameters
    -----------
       

    Returns
    -----------

    """ 
    # Generate data.
    X_df, labels_df = data_generator.generate(
                                                no_samples,
                                                npoints=no_sample_points,
                                                rescale=rescale
    )

    # Compute joint empirical distribution.
    X_df["joint"] = X_df['A'].apply(lambda x: x.tolist()) + X_df['B'].apply(lambda x: x.tolist())

    # Convert joint distribution into numpy array.
    X = onp.array(X_df["joint"].values.tolist())

    # Convert labels to numpy array. 
    labels = labels_df.to_numpy().reshape(no_samples, ).astype(onp.float32)
    
    # Remove joint distribution from pandas data frame.
    X_df.drop("joint", inplace=True, axis=1)

    return [X, X_df, labels]


def compute_model_results(
                            model: cdt.causality.pairwise,
                            data: onp.ndarray,
                            labels: onp.ndarray
    ) -> Dict:
    """ Computes the model predictions and scores.

    Parameters
    -----------
       

    Returns
    -----------

    """
    # Compute model predictions.
    predictions = onp.sign(model.predict(data)).astype(int)
    
    # Compute model scores.
    scores = (predictions == labels).astype(int)
    scores[scores==0] = -1
    
    return {"predictions": predictions, "scores": scores}


def return_max_entries(array: onp.ndarray):
    """ Returns the max entry in each row from a matrix as a vector.

    Parameters
    ----------- 
        array (onp.ndarray): [description]

    Returns
    ----------- 
        [type]: [description]
    """
    return array[range(len(array)), onp.argmax(onp.abs(array), axis=1)]


def normalise_decisions(decisions: onp.ndarray):
    """[summary]

    Parameters
    ----------- 
        decisions (onp.ndarray): [description]

    Returns
    ----------- 
        [type]: [description]
    """
    # Determine the sign of the decisions.
    decision_signs = onp.sign(decisions)
    
    # Compute the absolute value of the decisions.
    abs_decisions = onp.abs(decisions)
    
    # Normalise the absolute value of the decisions to 0, 1
    norm_abs_decisions = (abs_decisions - onp.min(abs_decisions))/(onp.max(abs_decisions) - onp.min(abs_decisions)) 
    
    # Return normalised decisions with the correct signs.
    return decision_signs * norm_abs_decisions


def generate_indices(
                        no_samples: int,
                        step_size: int
    ) -> list:
    """ Generates indices for the batched gram matrices.

    Description...
    Args:
        no_samples (int): [description]
        step_size (int): [description]

    Returns:
        indices: [description]
    """
    # Compute the index pairs for the upper triangular block matrices. 
    index_pairs = [[i*step_size, (i+1)*step_size] for i in range(int(no_samples/step_size))]

    indices = []

    no_pairs = len(index_pairs)

    for index_1 in range(no_pairs):
        for index_2 in range(index_1, no_pairs):
            indices.append(index_pairs[index_1] + index_pairs[index_2])

    return indices


def unpack_batched_train_grams(
                                    batched_grams: list,
                                    no_samples: int,
                                    indices: list
    ) -> jnp.ndarray:
    """[summary]

    """
    gram = onp.zeros((no_samples, no_samples))

    for i in range(len(indices)):
        index = indices[i]
        gram[index[0]:index[1], index[2]:index[3]] = batched_grams[i]
        gram[index[2]:index[3], index[0]:index[1]] = batched_grams[i].T

    return gram


def compute_batched_train_grams(
                            X_train: jnp.ndarray,
                            kernel: Callable,
                            params: Dict[str, float],
                            no_samples: int,
                            step_size: int,
                            n_jobs: int
    ):
    """[summary]

    Args:
        X_train (jnp.ndarray): [description]
        kernel (Callable): [description]
        params (Dict[str, float]): [description]
        no_samples (int): [description]
        step_size (int): [description]

    Returns:
        [type]: [description]
    """
    # Generate indices.
    indices = generate_indices(no_samples, step_size)

    # Compute batched gram matrices across CPU cores.
    batched_grams = Parallel(n_jobs=n_jobs)(delayed(compute_gram)\
                    (kernel, params, X_train[index[0]:index[1]],
                    X_train[index[2]:index[3]]) for index in indices)

    # Unpack the batched gram matrices.
    gram = unpack_batched_train_grams(batched_grams, no_samples, indices)

    return gram

def unpack_batched_test_gram(
                                batched_grams: list,
                                no_test_samples: int,
                                no_train_samples: int,
                                no_steps: int
    ) -> jnp.ndarray:
    """[summary]

    Args:
        batched_gram_matrices (list): [description]
        no_samples (int): [description]
        indices (list): [description]

    Returns:
        jnp.ndarray: [description]
    """
    gram = onp.zeros((no_test_samples, no_train_samples))

    for i in range(no_steps):
        gram[:,i:(i+1)*no_test_samples] = batched_grams[i]

    return gram

def compute_batched_test_gram(
                            X_test: jnp.ndarray,   
                            X_train: jnp.ndarray,                                                     
                            kernel: Callable,
                            params: Dict[str, float],
                            no_test_samples: int,                            
                            no_train_samples: int,
                            n_jobs: int
    ):
    """[summary]

    Args:
        X_train (jnp.ndarray): [description]
        X_test (jnp.ndarray): [description]        
        kernel (Callable): [description]
        params (Dict[str, float]): [description]
        no_samples (int): [description]
        step_size (int): [description]

    Returns:
        [type]: [description]
    """
    # Determine number of batched gram matrices we need to compute.
    no_steps = int(no_train_samples/no_test_samples)
    
    # Compute batched gram matrices across CPU cores.
    batched_gram = Parallel(n_jobs=n_jobs)(delayed(compute_gram)\
                    (kernel, params, X_test,
                    X_train[i:(i+1)*no_test_samples,:]) for i in range(no_steps))

    # Unpack the batched gram matrices.
    gram = unpack_batched_test_gram(batched_gram, no_test_samples, no_train_samples, no_steps)

    return gram


def smm_compute_model_results( smm, smm_name, data, labels):
    """ Fits the SMMs.
    
    Parameters
    -----------
       

    Returns
    -----------

    """
    # Fit the SMM with the training data.
    smm.compute_model_results(data, labels)
    
    return smm_name, smm


def smm_fit( smm, smm_name, gram_matrix):
    """ Fits the SMMs.
    
    Parameters
    -----------
       

    Returns
    -----------

    """
    # Fit the SMM with the training data.
    smm.fit(
                gram_matrix,
    )
    return smm_name, smm

    
def smm_predict( smm, smm_name, data, gram_matrix, labels):
    """ Predicts test data with the SMMs.
    
    Parameters
    -----------
       

    Returns
    -----------

    """
    # Use the SMM to predict the causal directions of the test data.
    smm.predict(
                data,
                gram_matrix,
                labels
    )
    return smm_name, smm

def smm_contribution(no_test_samples, ensemble_params, smm):
    """ Computes the contribution from each SMM to the ensemble. 
    
    Parameters
    -----------
       

    Returns
    -----------

    """
    # Initialise SMM contribution array.
    smm_contribution = onp.zeros((no_test_samples, ))
    
    # Store SMM decisions in temporary array.
    decisions = smm.decisions
    
    # Update decisions according to ensemble configuration.
    if ensemble_params["norm_decisions"]:
        decisions = normalise_decisions(decisions)
                
    if ensemble_params["exp_decision_function"]:
        decisions = onp.exp(decisions)
                
    # Compute SMM contribution according to ensemble configuration.
    if ensemble_params["accuracy"]:
        smm_contribution = decisions*smm.model_predictions*onp.power(smm.accuracy, ensemble_params["accuracy_exponent"])
    else:
        smm_contribution = decisions*smm.model_predictions

    return smm_contribution

