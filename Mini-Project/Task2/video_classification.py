#%%
import utils
import numpy as np
from sklearn.metrics import accuracy_score
from dimensionality_reduction import Quantizer
from sklearn.tree import DecisionTreeClassifier


# %%
class VideoClassifier:
    """Train a video classifier with decision tree
    """

    def __init__(self, quantizer: Quantizer, classifier):
        """Construct a classifier
        """
        pass
    
    def fit(self, X_train,y_train):
        """Train the classifier.
        """
        pass
    
    def predict(self, X) -> np.array:
        """Predict list of videos
        """
        pass