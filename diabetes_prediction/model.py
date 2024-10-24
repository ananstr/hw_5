# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class Model:
    def __init__(self, feature_columns, target_column, model_type='random_forest', **kwargs):
    # Initializes the Model class
    # Takes as arguments the list of feature columns, the target column name, the model type, and additional hyperparameters   
        self.feature_columns = feature_columns
        self.target_column = target_column
        
class Model:
    def __init__(self, model_type='random_forest', **kwargs):
    # Choose model based on type
        if model_type == 'logistic':
            self.model = LogisticRegression(**kwargs)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Model type '{model_type}' not supported.")

    def train(self, df):
    # Train the model on the data allocated for training
        X_train = df[self.feature_columns]
        y_train = df[self.target_column]
        self.model.fit(X_train, y_train)

    def predict(self, df):
    # Predict probabilities on the provided data
    # Takes a df as argument and returns predicted probabilities
        X_test = df[self.feature_columns]
        return self.model.predict_proba(X_test)[:, 1]

    def compute_roc_auc(self, df, y_true):
    # Compute ROC AUC score
    # Takes as arguments a df for prediction and true labels, and returns the ROC AUC score
        y_pred = self.predict(df)
        return roc_auc_score(y_true, y_pred)


