from utilities import compute_model_results, normalise_decisions
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer

class SMM():
    
    def __init__(self, model, model_name, svm_param_grid, verbose):
        """ SMM class.
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        self.model = model
        self.model_name = model_name
        self.svm_param_grid = svm_param_grid
        self.estimator = None
        self.decisions = None 
        self.accuracy = None 
        self.model_predictions = None
        self.model_accuracy = None
        self.y_train = None 
        self.verbose = verbose
        
    def compute_model_results(self, data_train, labels_train):
        
        if self.verbose:
            print("\nComputing predicitons for ", self.model_name)
            start_time = timer()
            
        model_results = compute_model_results(self.model, data_train, labels_train)
        
        # Record time taken if requested.
        if self.verbose:
            time_taken = timer() - start_time
            print("\nFinished computing predictions for ", self.model_name, "in ", time_taken, " seconds.")
            
        self.y_train = model_results["scores"]
        
    def fit(self, gram_matrix_train):
        """ Fits the SMM. 
        
        Parameters
        -----------
        

        Returns
        -----------

        """        
        # Fit SMM using grid search over the regularisation constant C.
        grid = GridSearchCV(
                                svm.SVC(kernel='precomputed'),
                                self.svm_param_grid,
                                refit=True
        )
      
        grid.fit(gram_matrix_train, self.y_train)
    
        # Store the best estimator.
        self.estimator = grid.best_estimator_
        
        
    def predict(self, data_test, gram_matrix_test, labels_test):
        """ Predicts test data with the SMM. 
        
        Parameters
        -----------
        

        Returns
        -----------

        """
        # Compute model predictions and scores on the test data.
        model_results = compute_model_results(self.model, data_test, labels_test)
        
        self.model_predictions = model_results["predictions"]
        
        y_test = model_results["scores"]
        
        # Compute SMM decision function on test data.
        self.decisions = self.estimator.decision_function(gram_matrix_test)
     
        # Compute SMM accuracy on test data.
        self.accuracy = self.estimator.score(gram_matrix_test, y_test)
     
        # Compute associated model accuracy on test data.
        self.model_accuracy = accuracy_score(labels_test, self.model_predictions)