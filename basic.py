import cdt 
import time
import jax.numpy as jnp
import numpy as np 
from sklearn import svm
from sklearn.metrics import accuracy_score
from utilities import compute_gram_matrix, rbf_kernel, rq_kernel, generate_data, compute_classifier_scores
from utilities import return_max_entries, normalise_decisions, compute_training_gram
from sklearn.model_selection import GridSearchCV
from utilities import ensemble_1, ensemble_2, ensemble_3, ensemble_4, ensemble_5
from utilities import ensemble_6, ensemble_7, ensemble_8, ensemble_9, ensemble_10
## Try on MAC!

no_train_samples = 50
no_sample_points = 200
no_test_samples = 50
step_size = 100

data_generator = cdt.data.CausalPairGenerator("sigmoid_mix", noise_coeff=0.2)


rescale = True

#X_train, X_train_df, labels_train = generate_data(data_generator, no_train_samples, no_sample_points, rescale)
#X_test, X_test_df, labels_test = generate_data(data_generator, no_test_samples, no_sample_points, rescale)

"""
X_train_, X_train_df_temp, labels_train_temp = generate_data(data_generator, no_train_samples, no_sample_points, rescale)

RCC = cdt.causality.pairwise.RCC()

RCC.fit(X_train_df_temp, labels_train_temp)
"""
## Add in RCC model that require training, train them on seperate 
## training data and then add them
## Create abstract class for a classifier and then 

print(type(cdt.causality.pairwise.RECI()))
quit()
classifiers = {
                "RECI": cdt.causality.pairwise.RECI(),
                "ANM" : cdt.causality.pairwise.ANM(),    
                "CDS" : cdt.causality.pairwise.CDS(),
                "IGCI" : cdt.causality.pairwise.IGCI(), 
                "BV" : cdt.causality.pairwise.BivariateFit(), 
}

models = {
                "RECI": [],
                "ANM" : [],    
                "CDS" : [],
                "IGCI": [], 
                "BV": [], 
}

kernel_params = {"gamma": no_train_samples, "alpha": 2}
kernel_params["factor"] = 0.5*kernel_params["gamma"]*(1/kernel_params["alpha"])

#kernel_params = {"gamma": no_train_samples}

X_train = jnp.asarray(X_train)
X_test = jnp.asarray(X_test)

"""
st = time.time()
gram_matrix_train = compute_training_gram(X_train, rbf_kernel, kernel_params, no_train_samples, step_size)
print("Batched 100: ", time.time() - st)
"""

st = time.time()
gram_matrix_train =  compute_gram(rq_kernel, kernel_params, X_train , X_train )
print("Non batched: ", time.time() - st)
quit()
gram_matrix_test = compute_gram(rq_kernel, kernel_params, X_test , X_train )

param_grid = {'C': np.linspace(1e-1, 1e3, 20)}

print("Grams computed ")
## We want this to be parallelised across cores as well.
support_indices = set()
for classifier_name, classifier in classifiers.items():
    
    start_time = time.time()
    _, y_train = compute_classifier_scores(classifier, X_train_df, labels_train)
    print("Classification time: ", time.time() - start_time)
    
    grid = GridSearchCV(svm.SVC(kernel='precomputed'),param_grid,refit=True)
    grid.fit(gram_matrix_train, y_train)
    
    model = grid.best_estimator_
    model_predictions, model_scores = compute_classifier_scores(classifier, X_test_df, labels_test)
    
    model_decisions = model.decision_function(gram_matrix_test)
    model_accuracy = model.score(gram_matrix_train, y_train)
    
    model_decisions = normalise_decisions(model_decisions)
    
    models[classifier_name].append(model_decisions)
    models[classifier_name].append(model_predictions)
    models[classifier_name].append(model_accuracy)
    
    print("----------------------------------")
    print(classifier_name, " accuracy = ", accuracy_score(labels_test, model_predictions))
    print(classifier_name, " smm accuracy = ", model_accuracy)
    print("----------------------------------")

    support_indices = set.union(support_indices, set(model.support_))

print(len(list(support_indices)))
#### Add this as argument to ensemble class
#

print("--------------------------------------------------")
print ("Weighted standard: ", ensemble_1(no_test_samples, labels_test, classifiers, models))
print ("Weighted exp df: ", ensemble_2(no_test_samples, labels_test, classifiers, models))
print ("Weighted accuracy: ", ensemble_3(no_test_samples, labels_test, classifiers, models))
print ("Weighted accuracy exp df: ", ensemble_10(no_test_samples, labels_test, classifiers, models))
print ("Weighted accuracy squared: ", ensemble_4(no_test_samples, labels_test, classifiers, models))
print("--------------------------------------------------")
print ("Max standard: ", ensemble_5(no_test_samples, labels_test, classifiers, models))
print ("Max exp df: ", ensemble_6(no_test_samples, labels_test, classifiers, models))
print ("Max accuracy: ", ensemble_7(no_test_samples, labels_test, classifiers, models))
print ("Max accuracy exp df: ", ensemble_9(no_test_samples, labels_test, classifiers, models))
print ("Max accuracy squared: ", ensemble_8(no_test_samples, labels_test, classifiers, models))
print("--------------------------------------------------")
