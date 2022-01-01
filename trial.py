#### TODO - Add voting. 

import arbitrator
import cdt
import numpy as onp
from os import cpu_count
from utilities import compute_gram_matrix, rq_kernel

data_params = {
                "causal_mechanism": "sigmoid_add",
                "noise_coeff": 0.2,
                "no_sample_points": 50,
                "no_train_samples": 50,                
                "no_test_samples": 50,
                "rescale": True
}

computation_params = {
                        "batched": False,
                        "step_size": 100,
                        "n_jobs": cpu_count() - 1
}

smm_params = {
                "svm_param_grid": {'C': onp.linspace(1e-1, 1e3, 10)},
                "norm_decisions": True
}
kernel_params = {"gamma": data_params["no_train_samples"], "alpha": 2}
kernel_params["factor"] = 0.5*kernel_params["gamma"]*(1/kernel_params["alpha"])
kernel_func = rq_kernel

base_models = {
                "RECI": cdt.causality.pairwise.RECI(),
                "ANM" : cdt.causality.pairwise.ANM(),    
                "CDS" : cdt.causality.pairwise.CDS(),
                "IGCI" : cdt.causality.pairwise.IGCI(), 
                "BV" : cdt.causality.pairwise.BivariateFit(), 
}

verbose = True

arb = arbitrator.Arbitrator(
                                data_params,
                                kernel_func,
                                kernel_params,
                                computation_params,
                                smm_params,
                                base_models,
                                verbose
)

arb.generate_train_data()
arb.generate_test_data()
arb.compute_training_gram_matrix()
arb.compute_testing_gram_matrix()
arb.initialise_smms()
arb.train_smms()
arb.predict_smms()