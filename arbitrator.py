import cdt
import jax.numpy as jnp
from smm import SMM
from joblib import Parallel, delayed
from utilities import generate_data, compute_batched_gram_matrix, compute_gram_matrix
from utilities import smm_fit, smm_predict


class Arbitrator():
    
    def  __init__(
                    self,
                    data_params,
                    kernel_func,
                    kernel_params,
                    computation_params,
                    smm_params,
                    base_models,
                    verbose
        ):
        
        self.data_params = data_params
        self.kernel_params = kernel_params
        self.kernel_func = kernel_func
        self.computation_params = computation_params
        self.smm_params = smm_params
        self.base_models = base_models
        self.verbose = verbose

        self.data_generator = None
        self.data_train = None
        self.data_test = None
        self.smms = None
        
        self.initialise_data_dictionaries()
        self.initialise_data_generator()
        
    def initialise_data_dictionaries(self):
        
        self.data_train = {"X": None, "X_df": None, "labels": None}
        self.data_test = {"X": None, "X_df": None, "labels": None}

    def initialise_data_generator(self):
            
        self.data_generator = cdt.data.CausalPairGenerator(
                            causal_mechanism=self.data_params["causal_mechanism"],
                            noise_coeff=self.data_params["noise_coeff"]
        )
        
    def initialise_smms(self):
        
        self.smms = {}
        
        for base_model_name, base_model in self.base_models.items():
            
            self.smms[base_model_name] = SMM(
                                        base_model,
                                        self.smm_params["svm_param_grid"],
                                        self.smm_params["norm_decisions"]
            )
                            
    def generate_train_data(self):
         
        training_data = generate_data(
                                        self.data_generator, 
                                        self.data_params["no_train_samples"],
                                        self.data_params["no_sample_points"],
                                        self.data_params["rescale"]
        )
    
        self.data_train["X"] = jnp.asarray(training_data[0])
        self.data_train["X_df"] = training_data[1]
        self.data_train["labels"] = training_data[2]
        
    def generate_test_data(self):
        
        testing_data = generate_data(
                                        self.data_generator, 
                                        self.data_params["no_test_samples"],
                                        self.data_params["no_sample_points"],
                                        self.data_params["rescale"]
        )
        
        self.data_test["X"] = jnp.asarray(testing_data[0])
        self.data_test["X_df"] = testing_data[1]
        self.data_test["labels"] = testing_data[2]
        
    def compute_training_gram_matrix(self):
        
        if self.computation_params["batched"]==True:
            self.data_train["gram_matrix"]  = compute_batched_gram_matrix(
                                self.data_train["X"],
                                self.kernel_func,
                                self.kernel_params,
                                self.data_params["no_train_samples"],
                                self.computation_params["step_size"],
                                self.computation_params["n_jobs"]  
            )
        else:
            self.data_train["gram_matrix"] = compute_gram_matrix(
                                self.kernel_func,
                                self.kernel_params,
                                self.data_train["X"],
                                self.data_train["X"],
            )
            
    ### Add batched testing if necessary.
    def compute_testing_gram_matrix(self):
        
        self.data_test["gram_matrix"]  = compute_gram_matrix(
                                self.kernel_func,
                                self.kernel_params,
                                self.data_test["X"],
                                self.data_train["X"]
        )
        
    def train_smms(self):
        
        smms_data = Parallel(n_jobs=self.computation_params["n_jobs"])\
                            (delayed(smm_fit)(
                                    smm, 
                                    smm_name,
                                    self.data_train["X_df"],
                                    self.data_train["gram_matrix"],
                                    self.data_train["labels"]) \
                                for smm_name, smm in self.smms.items()
        )
      
        for smm_data in smms_data:
            self.smms[smm_data[0]] = smm_data[1]
                
    def predict_smms(self):
        
        smms_data = Parallel(n_jobs=self.computation_params["n_jobs"])\
                            (delayed(smm_predict)(
                                    smm, 
                                    smm_name,
                                    self.data_test["X_df"],
                                    self.data_test["gram_matrix"],
                                    self.data_test["labels"]) \
                                for smm_name, smm in self.smms.items()
        )
                            
        for smm_data in smms_data:
            
            self.smms[smm_data[0]] = smm_data[1]
            
            if self.verbose:
                print("------------------------------------------")
                print(smm_data[0], " model accuracy: ", smm_data[1].model_accuracy)
                print(smm_data[0], " SMM accuracy: ", smm_data[1].accuracy)

    def compute_ensemble_prediction(self):
        
        pass
    
