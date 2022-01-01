import functools
from re import A
import jax
import jax.numpy as jnp
import numpy as onp
import cdt
from typing import Callable, Dict 
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score



@functools.partial(jax.jit, static_argnums=(0, 4))
def pdist_squareform(no_sample_points: float, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """ Computes the squared euclidean distance matrix.

    This function computes the squared euclidean distance matrix between two 
    samples x, y from probability distributions P and Q.
    """
    I = jnp.ones((no_sample_points, ))
    return jnp.outer(jnp.power(x, 2), I) + jnp.outer(I, jnp.power(y, 2)) - 2*jnp.outer(x, y) 

@functools.partial(jax.jit,  static_argnums=(0, 4))
def rbf_kernel(
                no_sample_points: float,
                params: Dict[str, float],
                x: jnp.ndarray,
                y: jnp.ndarray
    ) -> jnp.ndarray:
    """ Computes the inner product between two samples x, y constructed from
    RBF kernel.
    
    This function computes the inner product between two samples x, y 
    from the embedded probability distributions mu(P)_k and mu(Q)_k. The
    kernel of the RKHS is the RBF kernel. 
    """
    return jnp.mean(jnp.exp(-params["gamma"] * pdist_squareform(no_sample_points, x, y)))

@functools.partial(jax.jit,  static_argnums=(0, 4))
def rq_kernel(
                no_sample_points: float,
                params: Dict[str, float],
                x: jnp.ndarray,
                y: jnp.ndarray
    ) -> jnp.ndarray:
    """ Computes the inner product between two samples x, y in a RKHS constructed from 
    RQ kernel.
    
    This function computes the inner product between two samples x, y 
    from the embedded probability distributions mu(P)_k and mu(Q)_k. The
    kernel of the RKHS is the RQ kernel. 
    """    
    return jnp.mean(jnp.power(1 + (params["factor"]\
                             *pdist_squareform(no_sample_points, x, y)), -params["alpha"]))

@functools.partial(jax.jit, static_argnums=(0, 4))
def compute_gram_matrix(
                    func: Callable,
                    params: Dict[str, float],
                    x: jnp.ndarray,
                    y: jnp.ndarray,
    ) -> jnp.ndarray:
    """ Computes the gram matrix between n samples x, y. 
    
    This function computes the gram matrix between n samples x, y from 
    the embedded probability distributions mu(P)_k and mu(Q)_k.
    """    
    # Determine the number of points in each sample.
    no_sample_points = x.shape[1]
    
    # Define the inner loop i.e. k(x_i, y_j) for all y_j.
    inner_loop = lambda x_1: jax.vmap(lambda y_1: func(no_sample_points, params, x_1, y_1))(y)
    
    # Compute the outer loop, i.e. the inner loop for all x_i.
    gram_matrix = jax.lax.map(inner_loop, x)

    return gram_matrix


def generate_data(
                    data_generator: cdt.data.causal_pair_generator.CausalPairGenerator,
                    no_samples: int,
                    no_sample_points: int,
                    rescale: bool
    ) -> list:
    """ Generates data. 
    
    This function generates data ... 
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

def compute_model_scores(
                            model: cdt.causality.pairwise,
                            data: onp.ndarray,
                            labels: onp.ndarray
    ) -> Dict:
    """ Computes the model predictions and scores.

    Parameters
    -----------
        classifier (cdt.causality.pairwise): [description]
        data (onp.ndarray): [description]
        labels (onp.ndarray): [description]

    Returns
    -----------

        Dict: [description]
    """
    # Compute model predictions.
    predictions = onp.sign(model.predict(data)).astype(int)
    
    # Compute model scores.
    scores = (predictions == labels).astype(int)
    scores[scores==0] = -1
    
    return {"Predictions": predictions, "Scores": scores}

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


def compute_batch_gram_matrix(
                                kernel: Callable,
                                params: Dict[str, float],
                                x: jnp.ndarray,
                                y: jnp.ndarray,
    ) -> jnp.ndarray:
    """ Computes batched gram matrix.
    
    Description...

    Args:
        kernel (Callable): [description]
        params (Dict[str, float]): [description]
        x (jnp.ndarray): [description]
        y (jnp.ndarray): [description]

    Returns:
        jnp.ndarray: [description]
    """
    gram_matrix = compute_gram_matrix(kernel, params, x , y )

    return gram_matrix


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


def unpack_batched_gram_matrices(
                                    batched_gram_matrices: list,
                                    no_samples: int,
                                    indices: list
    ) -> jnp.ndarray:
    """[summary]

    Args:
        batched_gram_matrices (list): [description]
        no_samples (int): [description]
        indices (list): [description]

    Returns:
        jnp.ndarray: [description]
    """
    gram_matrix = onp.zeros((no_samples, no_samples))

    for i in range(len(indices)):
        index = indices[i]
        gram_matrix[index[0]:index[1], index[2]:index[3]] = batched_gram_matrices[i]
        gram_matrix[index[2]:index[3], index[0]:index[1]] = batched_gram_matrices[i].T

    return gram_matrix


def compute_training_gram(
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
    batched_gram_matrices = Parallel(n_jobs=n_jobs)(delayed(compute_batch_gram_matrix)\
                    (kernel, params, X_train[index[0]:index[1]],
                    X_train[index[2]:index[3]]) for index in indices)

    # Unpack the batched gram matrices.
    gram_matrix = unpack_batched_gram_matrices(batched_gram_matrices, no_samples, indices)

    return gram_matrix



def ensemble_1(no_test_samples, labels_test, classifiers, models):
    
    meta_predictions = onp.zeros((no_test_samples, ))

    for classifier_name in classifiers.keys():

        decisions = models[classifier_name][0]
        predictions = models[classifier_name][1]
        accuracy = models[classifier_name][2]
        
        
        meta_predictions += decisions*predictions
        
    meta_predictions = onp.sign(meta_predictions)

    return accuracy_score(labels_test, meta_predictions)

def ensemble_2(no_test_samples, labels_test, classifiers, models):
    
    meta_predictions = onp.zeros((no_test_samples, ))

    for classifier_name in classifiers.keys():

        decisions = models[classifier_name][0]
        predictions = models[classifier_name][1]
        accuracy = models[classifier_name][2]
      
        meta_predictions += onp.exp(decisions)*predictions
        
    meta_predictions = onp.sign(meta_predictions)

    return accuracy_score(labels_test, meta_predictions)

def ensemble_3(no_test_samples, labels_test, classifiers, models):
    
    meta_predictions = onp.zeros((no_test_samples, ))

    for classifier_name in classifiers.keys():

        decisions = models[classifier_name][0]
        predictions = models[classifier_name][1]
        accuracy = models[classifier_name][2]
      
        meta_predictions += decisions*predictions*onp.power(accuracy, 1)
        
    meta_predictions = onp.sign(meta_predictions)

    return accuracy_score(labels_test, meta_predictions)

def ensemble_4(no_test_samples, labels_test, classifiers, models):
    
    meta_predictions = onp.zeros((no_test_samples, ))

    for classifier_name in classifiers.keys():

        decisions = models[classifier_name][0]
        predictions = models[classifier_name][1]
        accuracy = models[classifier_name][2]
      
        meta_predictions += decisions*predictions*onp.power(accuracy, 2)
        
    meta_predictions = onp.sign(meta_predictions)

    return accuracy_score(labels_test, meta_predictions)

def ensemble_5(no_test_samples, labels_test, classifiers, models):
    
    meta_predictions = onp.zeros((no_test_samples, len(classifiers) ))

    count = 0
    for classifier_name in classifiers.keys():

        decisions = models[classifier_name][0]
        predictions = models[classifier_name][1]
        accuracy = models[classifier_name][2]
      
        meta_predictions[:, count] = decisions*predictions
        
        count += 1
    meta_predictions = onp.sign(return_max_entries(meta_predictions))

    return accuracy_score(labels_test, meta_predictions)

def ensemble_6(no_test_samples, labels_test, classifiers, models):
    
    meta_predictions = onp.zeros((no_test_samples, len(classifiers) ))

    count = 0
    for classifier_name in classifiers.keys():

        decisions = models[classifier_name][0]
        predictions = models[classifier_name][1]
        accuracy = models[classifier_name][2]
      
        meta_predictions[:, count] = onp.exp(decisions)*predictions
        
        count += 1
    meta_predictions = onp.sign(return_max_entries(meta_predictions))

    return accuracy_score(labels_test, meta_predictions)

def ensemble_7(no_test_samples, labels_test, classifiers, models):
    
    meta_predictions = onp.zeros((no_test_samples, len(classifiers) ))

    count = 0
    for classifier_name in classifiers.keys():

        decisions = models[classifier_name][0]
        predictions = models[classifier_name][1]
        accuracy = models[classifier_name][2]
      
        meta_predictions[:, count] = decisions*predictions*onp.power(accuracy, 1)
        
        count += 1
        
    meta_predictions = onp.sign(return_max_entries(meta_predictions))

    return accuracy_score(labels_test, meta_predictions)

def ensemble_8(no_test_samples, labels_test, classifiers, models):
    
    meta_predictions = onp.zeros((no_test_samples, len(classifiers) ))

    count = 0
    for classifier_name in classifiers.keys():

        decisions = models[classifier_name][0]
        predictions = models[classifier_name][1]
        accuracy = models[classifier_name][2]
      
        meta_predictions[:, count] = decisions*predictions*onp.power(accuracy, 2)
        
        count += 1
        
    meta_predictions = onp.sign(return_max_entries(meta_predictions))

    return accuracy_score(labels_test, meta_predictions)

def ensemble_9(no_test_samples, labels_test, classifiers, models):
    
    meta_predictions = onp.zeros((no_test_samples, len(classifiers) ))

    count = 0
    for classifier_name in classifiers.keys():

        decisions = models[classifier_name][0]
        predictions = models[classifier_name][1]
        accuracy = models[classifier_name][2]
      
        meta_predictions[:, count] = onp.exp(decisions)*predictions*onp.power(accuracy, 1)
        
        count += 1
        
    meta_predictions = onp.sign(return_max_entries(meta_predictions))

    return accuracy_score(labels_test, meta_predictions)

def ensemble_10(no_test_samples, labels_test, classifiers, models):
    
    meta_predictions = onp.zeros((no_test_samples, ))

    for classifier_name in classifiers.keys():

        decisions = models[classifier_name][0]
        predictions = models[classifier_name][1]
        accuracy = models[classifier_name][2]
      
        meta_predictions += onp.exp(decisions)*predictions*onp.power(accuracy, 1)
        
    meta_predictions = onp.sign(meta_predictions)

    return accuracy_score(labels_test, meta_predictions)
