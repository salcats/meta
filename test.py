import cdt
import numpy as onp
from os import cpu_count
from arbitrator import Arbitrator
from utilities import rq_kernel, rbf_kernel

# Define data parameters.
data_params = {
                "causal_mechanism": "sigmoid_mix",
                "noise_coeff": 0.2,
                "no_sample_points": 250,
                "no_train_samples": 250,                
                "no_test_samples": 100,
                "rescale": True
}

# Define computation parameters.
computation_params = {
                        "batched": True,
                        "training_step_size": 50,
                        "n_jobs": cpu_count() - 1
}

# Define SMM parameters.
smm_params = {
                "svm_param_grid": {'C': onp.linspace(1e-1, 1e3, 20)},
}

# Define kernel parameters.
kernel_params = {"gamma": data_params["no_train_samples"]}
kernel_func = rbf_kernel

# Define base models. (Only using models that don't require training)
base_models = {
                "RECI": cdt.causality.pairwise.RECI(),
                "ANM" : cdt.causality.pairwise.ANM(),    
                "CDS" : cdt.causality.pairwise.CDS(),
                "IGCI" : cdt.causality.pairwise.IGCI(), 
                "BV" : cdt.causality.pairwise.BivariateFit(), 
}

# Determine verbosity.
verbose = True

kernel_params = {"gamma": data_params["no_train_samples"]}
kernel_func = rbf_kernel

# Initialise arbitrator.
arbitrator = Arbitrator(
                            data_params,
                            kernel_func,
                            kernel_params,
                            computation_params,
                            smm_params,
                            base_models,
                            verbose
)

run_hp_optimisation = False

if not run_hp_optimisation:
    
    # Run without kernel hyperparameter optimisation

    best_results_dataframe, best_accuracy = arbitrator.run_arbitration(run_hp_optimisation)

    print("\n")
    print(best_results_dataframe)

else:

    # Run without kernel hyperparameter optimisation

    hp_optimisation_params = {
                                "no_samples": 50,
                                "gamma_min": 1, 
                                "gamma_max": 1e4, 
                                "max_evals": 10
    }

    best_results_dataframe, best_accuracy = arbitrator.run_arbitration(
                                                        run_hp_optimisation,
                                                        hp_optimisation_params=hp_optimisation_params
    )
    print("\n")
    print(best_results_dataframe)
