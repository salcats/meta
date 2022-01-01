import cdt 
from utilities import compute_model_results, normalise_decisions
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score

class SMM():
    
    def __init__(self, model, svm_param_grid, norm_decisions):
        
        self.model = model
        self.svm_param_grid = svm_param_grid
        self.norm_decisions = norm_decisions
        self.estimator = None
        self.decisions = None 
        self.accuracy = None 
        self.predictions = None
        
    def fit(self, data_train, gram_matrix_train, labels_train):
        
        model_results = compute_model_results(self.model, data_train, labels_train)
        
        y_train = model_results["scores"]
        
        grid = GridSearchCV(
                                svm.SVC(kernel='precomputed'),
                                self.svm_param_grid,
                                refit=True
        )
      
        grid.fit(gram_matrix_train, y_train)
    
        self.estimator = grid.best_estimator_
        
        
    def predict(self, data_test, gram_matrix_test, labels_test):
        
        model_results = compute_model_results(self.model, data_test, labels_test)
        
        self.predictions = model_results["predictions"]
        
        y_test = model_results["scores"]
        
        self.decisions = self.estimator.decision_function(gram_matrix_test)
        
        if self.norm_decisions:
            self.decisions = normalise_decisions(self.decisions)
        
        self.accuracy = self.estimator.score(gram_matrix_test, y_test)
        self.model_accuracy = accuracy_score(labels_test, self.predictions)