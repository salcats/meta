import cdt
import jax.numpy as jnp
import pandas as pd
import itertools
from smm import SMM
from joblib import Parallel, delayed
from ensembler import Ensembler
from utilities import generate_data, compute_batched_train_grams, compute_batched_test_gram, compute_gram
from utilities import smm_fit, smm_predict, smm_compute_model_results
from timeit import default_timer as timer
from hyperopt import hp, fmin, tpe


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
        """  Arbitrator class.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """
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
        self.ensembler = None 
        
        # Initialise the train, test data dictionaries.
        self.initialise_data_dictionaries()
        
        # Initialise the data generator.
        self.initialise_data_generator()
        
    def initialise_data_dictionaries(self):
        """  Initialise data dictionaries.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        self.data_train = {"X": None, "X_df": None, "labels": None}
        self.data_test = {"X": None, "X_df": None, "labels": None}

    def initialise_data_generator(self):
        """  Initialise data generator.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """    
        self.data_generator = cdt.data.CausalPairGenerator(
                            causal_mechanism=self.data_params["causal_mechanism"],
                            noise_coeff=self.data_params["noise_coeff"]
        )
        
    def initialise_smms(self):
        """  Initialise SMMs.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        # Initialise dictionary.
        self.smms = {}
        
        # Loop through base models.
        for base_model_name, base_model in self.base_models.items():
            
            # Create SMM object for each base model.
            self.smms[base_model_name] = SMM(
                                        base_model,
                                        self.smm_params["svm_param_grid"],
            )
         
    def generate_train_data(self):
        """  Generate training data.   

        """     
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
        """  Generate testing data.   

        """ 
        testing_data = generate_data(
                                        self.data_generator, 
                                        self.data_params["no_test_samples"],
                                        self.data_params["no_sample_points"],
                                        self.data_params["rescale"]
        )
        
        self.data_test["X"] = jnp.asarray(testing_data[0])
        self.data_test["X_df"] = testing_data[1]
        self.data_test["labels"] = testing_data[2]
        
    def compute_train_gram(self):
        """  Compute training gram matrix.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        
        if self.verbose:
            print("\nComputing training gram matrix.")
            start_time = timer()
            
        # Compute training gram matrix.
        
        if self.computation_params["batched"]==True:
            self.data_train["gram"]  = compute_batched_train_grams(
                                self.data_train["X"],
                                self.kernel_func,
                                self.kernel_params,
                                self.data_params["no_train_samples"],
                                self.computation_params["training_step_size"],
                                self.computation_params["n_jobs"]  
            )
        else:
            self.data_train["gram"] = compute_gram(
                                self.kernel_func,
                                self.kernel_params,
                                self.data_train["X"],
                                self.data_train["X"],
            )
        
        # Record time taken if requested.
        if self.verbose:
            time_taken = timer() - start_time
            print("\nFinished computing training gram matrix in ", time_taken, " seconds.")
            
    def compute_test_gram(self):
        """  Compute testing gram matrix.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        if self.verbose:
            print("\nComputing testing gram matrix.")
            start_time = timer()
            
        # Compute testing gram matrix.
        if self.computation_params["batched"]==True:
            self.data_test["gram"]  = compute_batched_test_gram(
                                self.data_test["X"],
                                self.data_train["X"],
                                self.kernel_func,
                                self.kernel_params,
                                self.data_params["no_test_samples"],
                                self.data_params["no_train_samples"],
                                self.computation_params["n_jobs"]  
            )
        else:
            self.data_test["gram"]  = compute_gram(
                                    self.kernel_func,
                                    self.kernel_params,
                                    self.data_test["X"],
                                    self.data_train["X"]
            )
            
        # Record time taken if requested.
        if self.verbose:
            time_taken = timer() - start_time
            print("\nFinished computing testing gram matrix in ", time_taken, " seconds.")   
         
    def compute_model_results_smms(self):
        """  Computes the model results for each SMM.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        if self.verbose:
            print("Computing model results for each SMM.")
            start_time = timer()
            
        # Compute model results for each SMM in parallel across CPU cores.
        smms_data = Parallel(n_jobs=self.computation_params["n_jobs"])\
                            (delayed(smm_compute_model_results)(
                                    smm, 
                                    smm_name,
                                    self.data_train["X_df"],
                                    self.data_train["labels"]) \
                                for smm_name, smm in self.smms.items()
        )
      
        # Store results back in dictionary.
        for smm_data in smms_data:
            self.smms[smm_data[0]] = smm_data[1]
        
        # Record time taken if requested.
        if self.verbose:
            time_taken = timer() - start_time
            print("\nFinished computing SMM model results in ", time_taken, " seconds.")
                  
    def fit_smms(self):
        """  Fit SMMs.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        if self.verbose:
            print("\nTraining SMMs.")
            start_time = timer()
            
        # Fit SMMs in parallel across CPU cores.
        smms_data = Parallel(n_jobs=self.computation_params["n_jobs"])\
                            (delayed(smm_fit)(
                                    smm, 
                                    smm_name,
                                    self.data_train["gram"]) \
                                for smm_name, smm in self.smms.items()
        )
      
        # Store results back in dictionary.
        for smm_data in smms_data:
            self.smms[smm_data[0]] = smm_data[1]
        
        # Record time taken if requested.
        if self.verbose:
            time_taken = timer() - start_time
            print("\nFinished training SMMs in ", time_taken, " seconds.")
            
    def predict_smms(self):
        """  Predict test data with SMMs.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        if self.verbose:
            print("\nPredicting with SMMs.")
            start_time = timer()

        # Predict test data with SMMs in parallel across CPU cores.
        smms_data = Parallel(n_jobs=self.computation_params["n_jobs"])\
                            (delayed(smm_predict)(
                                    smm, 
                                    smm_name,
                                    self.data_test["X_df"],
                                    self.data_test["gram"],
                                    self.data_test["labels"]) \
                                for smm_name, smm in self.smms.items()
        )
        
        # Store results back in dictionary.    
        for smm_data in smms_data:
            
            self.smms[smm_data[0]] = smm_data[1]
            
            # Record results if requested.
            if self.verbose:
                print("------------------------------------------")
                print(smm_data[0], " model accuracy: ", smm_data[1].model_accuracy)
                print(smm_data[0], " SMM accuracy: ", smm_data[1].accuracy)
        
        print("------------------------------------------")
        
        # Record time taken if requested.
        if self.verbose:
            time_taken = timer() - start_time
            print("\nFinished predicting with SMMs in ", time_taken, " seconds.")
                
    def initialise_ensembler(self):
        """  Initialise ensembler with SMMs.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        # Initialise ensembler.
        self.ensembler = Ensembler(self.smms, self.data_test)
        
        # Determine the number of test samples.
        self.ensembler.determine_no_test_samples()
        
    def compute_ensemble_prediction(self, ensemble_params):
        """  Compute ensemble prediction.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        if ensemble_params["average"]:
            
            # If average ensemble requested compute average ensemble.
            ensemble_accuracy = self.ensembler.average_ensemble(ensemble_params)
        
        else:
            
            # If maximum ensemble requested compute average ensemble.
            ensemble_accuracy = self.ensembler.maximum_ensemble(ensemble_params)

        return ensemble_accuracy
    
    def compute_all_ensembles(self, max_accuracy_exponent):
        """ Computes all possible ensemble configurations. 
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        # Initialise ensemble parameter dictionary.
        ensemble_params_dict = {
                                    "Average": [],
                                    "Maximum": [],
                                    "Normalised decisions": [],
                                    "Exponentiated decisions": [],
                                    "Accuracy weighted": [],
                                    "Accuracy exponent": [],
                                    "Ensemble accuracy": []
        }

        # Compute all ensembel configurations for boolean valued ensemble parameters.
        ensemble_configurations = ["".join(seq) for seq in itertools.product("01", repeat=4)]

        # Compute ensemble accuracy for each configuration.
        for configuration in ensemble_configurations:
            
            # Loop through accuracy powers.
            for accuracy_exponent in range(1, max_accuracy_exponent+1):
                
                # Store ensemble parameters for this configuration.
                ensemble_params = {
                                    "average": bool(int(configuration[0])),
                                    "norm_decisions": bool(int(configuration[1])),
                                    "exp_decision_function": bool(int(configuration[2])),
                                    "accuracy": bool(int(configuration[3])),
                                    "accuracy_exponent": accuracy_exponent
                }
                
                # Compute and store ensemble accuracy.
                ensemble_params_dict["Ensemble accuracy"].append(self.compute_ensemble_prediction(ensemble_params))

                # Store ensemble configuration. 
                ensemble_params_dict["Average"].append(ensemble_params["average"])
                ensemble_params_dict["Maximum"].append(not(ensemble_params["average"]))
                ensemble_params_dict["Normalised decisions"].append(ensemble_params["norm_decisions"])
                ensemble_params_dict["Exponentiated decisions"].append(ensemble_params["exp_decision_function"])
                ensemble_params_dict["Accuracy weighted"].append(ensemble_params["accuracy"])
                ensemble_params_dict["Accuracy exponent"].append(ensemble_params["accuracy_exponent"])

        # Convert dict to dataframe.
        results_dataframe = pd.DataFrame(data=ensemble_params_dict)

        # Compute best accuracy.
        best_accuracy = results_dataframe["Ensemble accuracy"].max()
        
        # Extract the best ensemble configurations.
        best_results_dataframe = results_dataframe[results_dataframe["Ensemble accuracy"] == best_accuracy]

        return best_results_dataframe, best_accuracy
    
    def perform_kernel_hp_optimisation(self, hp_optimisation_params):
        
        # Store number of training samples in temporary variable.
        no_train_samples = self.data_params["no_train_samples"]
        
        # Store batched request in temporary variable.
        batched = self.computation_params["batched"]
        
        # Store verbosity.
        verbose = self.verbose
        
        # Change these for the hp optimisation.
        self.data_params["no_train_samples"] = hp_optimisation_params["no_samples"]
        self.computation_params["batched"] = False
        
        self.verbose = False
        
        self.generate_train_data()
        self.generate_test_data()
        self.compute_model_results_smms()
        
        parameter_space = {"gamma": hp.uniform(
                                                "gamma",
                                                hp_optimisation_params["gamma_min"], 
                                                hp_optimisation_params["gamma_max"]
                                            )
        }
        
        print("\nPerforming hyper parameter optimisation for the kernel parameters.")
        best = fmin(
                        self.kernel_hp_objective,
                        parameter_space,
                        algo=tpe.suggest,
                        max_evals=hp_optimisation_params["max_evals"]
        )
        print("\nFinished performing hyper parameter optimisation for the kernel parameters.")

        self.data_params["no_train_samples"] = no_train_samples
        self.computation_params["batched"] = batched
        self.kernel_params["gamma"] = best["gamma"]
        self.verbose = verbose
 
    def kernel_hp_objective(self, parameter_space):

        kernel_params = {"gamma": parameter_space["gamma"]}
        
        self.kernel_params = kernel_params
                
        # Compute gram matrices.
        self.compute_train_gram()
        self.compute_test_gram()
                
        # Fit and predict with SMMs
        self.fit_smms()
        self.predict_smms()

        # Initialise ensembler
        self.initialise_ensembler()

        # Compute all ensemble configurations and return the best results.
        max_accuracy_exponent = 4
        _, best_accuracy = self.compute_all_ensembles(max_accuracy_exponent)
        
        return 1 - best_accuracy
    
    def run_arbitration(self, run_hp_optimisation, hp_optimisation_params=None):

        self.initialise_smms()

        if run_hp_optimisation:

            self.perform_kernel_hp_optimisation(hp_optimisation_params)


        # Generate data.
        self.generate_train_data()
        self.generate_test_data()
        self.compute_model_results_smms()

        # Compute gram matrices.
        self.compute_train_gram()
        self.compute_test_gram()
                
        # Initialise, fit and predict with SMMs
        self.fit_smms()
        self.predict_smms()

        # Initialise ensembler
        self.initialise_ensembler()

        # Compute all ensemble configurations and return the best results.
        max_accuracy_exponent = 4
        best_results_dataframe, best_accuracy = self.compute_all_ensembles(max_accuracy_exponent)

        return best_results_dataframe, best_accuracy

