# %%
# Import classes from the diabetes_prediction package
from diabetes_prediction.data_loader import DataLoader
from diabetes_prediction.data_preprocessor import NaNRemover, NaNFiller
from diabetes_prediction.feature_extractor import GenderBinaryTransformer, EthnicityOneHotEncoder
from diabetes_prediction.model import Model

# Load the data using the DataLoader class
file_path = 'sample_diabetes_mellitus_data.csv'
data_loader = DataLoader(file_path)
train_df, test_df = data_loader.load_and_split()

# Preprocess the data using the NaNRemover and NaNFiller classes
# Remove NaNs in 'age', 'gender', 'ethnicity'
nan_remover = NaNRemover(columns_with_nan=['age', 'gender', 'ethnicity'])
train_df = nan_remover.remove_nan(train_df)
test_df = nan_remover.remove_nan(test_df)

# Fill NaNs in 'height', 'weight'
nan_filler = NaNFiller(columns_to_fill=['height', 'weight'])
train_df = nan_filler.fill_nan(train_df)
test_df = nan_filler.fill_nan(test_df)

# Extract features using the GenderBinaryTransformer and EthnicityOneHotEncoder classes
gender_transformer = GenderBinaryTransformer()
ethnicity_transformer = EthnicityOneHotEncoder()

train_df = gender_transformer.transform(train_df)
test_df = gender_transformer.transform(test_df)

train_df = ethnicity_transformer.transform(train_df)
test_df = ethnicity_transformer.transform(test_df)

# Instantiate Model class
feature_columns = ['age', 'height', 'weight', 'aids', 'cirrhosis', 'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']
target_column = 'diabetes_mellitus'

model = Model(feature_columns=feature_columns, target_column=target_column, model_type='logistic')

# Train the model
model.train(train_df)

# Predict probabilities on the test set
test_df['predictions'] = model.predict(test_df)

# Compute ROC AUC score
roc_auc = model.compute_roc_auc(test_df, test_df[target_column])
print(f"Test ROC-AUC: {roc_auc}")


