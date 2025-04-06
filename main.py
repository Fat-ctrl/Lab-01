from metaflow import FlowSpec, step, card
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import datasets

class MLFlowPipeline(FlowSpec):
    
    @card
    @step
    def start(self):
        self.model_configs = [
            ('dt', DecisionTreeClassifier(max_depth=4)),
            ('knn', KNeighborsClassifier(n_neighbors=7)),
            ('svc', SVC(kernel='rbf', probability=True))
        ]
        self.next(self.load_data)

    @card
    @step
    def load_data(self):
        # Load Iris dataset
        iris = datasets.load_iris()
        X = iris.data  # pylint: disable=no-member
        y = iris.target  # pylint: disable=no-member
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.next(self.train_models, foreach='model_configs')

    @card
    @step
    def train_models(self):
        self.trained_models = []
        name, model = self.input
        model.fit(self.X_train, self.y_train)
        acc = accuracy_score(self.y_val, model.predict(self.X_val))
        self.trained_models.append((name, model, acc))
        self.next(self.join)
    
    @card    
    @step
    def join(self, inputs):
        self.trained_models = [input.trained_models for input in inputs]
        self.trained_models = [item for sublist in self.trained_models for item in sublist]  # Flatten the list
        # Ensure variables are passed to the next step
        self.X_train = inputs[0].X_train  # Assuming all steps have the same data
        self.X_val = inputs[0].X_val
        self.y_train = inputs[0].y_train
        self.y_val = inputs[0].y_val
        self.next(self.ensemble)

    @card
    @step
    def ensemble(self):
        # Ensure the variables are correctly passed to the ensemble step
        voting_clf = VotingClassifier(estimators=[(name, model) for name, model, _ in self.trained_models],
                                      voting='soft')
        voting_clf.fit(self.X_train, self.y_train)
        y_pred = voting_clf.predict(self.X_val)
        final_accuracy = accuracy_score(self.y_val, y_pred)
        print(f"Final Soft Voting Accuracy: {final_accuracy:.4f}")
        self.next(self.end)
    @card    
    @step
    def end(self):
        print("Pipeline completed.")

if __name__ == '__main__':
    MLFlowPipeline()
