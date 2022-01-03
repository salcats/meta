import numpy as onp
from utilities import smm_contribution, return_max_entries
from sklearn.metrics import accuracy_score


class Ensembler():

    def __init__(self, smms, data_test):
        """ Class for implementing an ensemble scheme with the collection
        of SMMs. 
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        self.smms = smms
        self.data_test = data_test
        self.no_test_samples = None
        
    def determine_no_test_samples(self):
        """  Determine the number of test samples.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        self.no_test_samples = len(self.data_test["labels"])
        
    def average_ensemble(self, ensemble_params):
        """  Compute average ensemble.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """ 
        # Initialise ensemble predictions array.
        ensemble_predictions = onp.zeros((self.no_test_samples, ))

        # Loop through individual SMMs.
        for smm in self.smms.values():
            
            # Compute contribution from each SMM.
            ensemble_predictions += smm_contribution(
                                                    self.no_test_samples,
                                                    ensemble_params,
                                                    smm
            )
        
        # Determine ensemble causal prediction and return the accuracy.                   
        ensemble_predictions = onp.sign(ensemble_predictions)

        return accuracy_score(self.data_test["labels"], ensemble_predictions)
    
    def maximum_ensemble(self, ensemble_params):
        """  Compute maximum ensemble.      
        
        Parameters
        -----------
        

        Returns
        -----------

        """   
        # Initialise ensemble predictions array. 
        ensemble_predictions = onp.zeros((self.no_test_samples, len(self.smms)))

        # Loop through individual SMMs.
        for index, smm in enumerate(self.smms.values()):
    
            # Compute contribution from each SMM.
            ensemble_predictions[:, index] = smm_contribution(
                                                    self.no_test_samples,
                                                    ensemble_params,
                                                    smm
            )

        # Determine ensemble causal prediction and return the accuracy.                   
        ensemble_predictions = onp.sign(return_max_entries(ensemble_predictions))

        return accuracy_score(self.data_test["labels"], ensemble_predictions)
    
