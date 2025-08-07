import utils
import helper_code
import feature_eda_pipeline_GPT as featurize

records = utils.prepare_stratification(helper_code.find_records_abs("../training_data"))

records_paths = [record['record'] for record in records]
df, error_records = featurize.run_feature_extraction_for_records_absolute(records_paths, "feature_importance_ranking.csv", 20, -1)

# Convert the list of dictionaries to a DataFrame
processed_df = pd.DataFrame(processed_records)

# Extract the features DataFrame from the tuple returned by the function
features_df = df[0]

# --- Merge the DataFrames ---
# Ensure the 'exam_id' columns are of the same type (string) for a clean merge
features_df['exam_id'] = features_df['exam_id'].astype(str)
processed_df['exam_id'] = processed_df['exam_id'].astype(str)

# Perform a left merge to add SNOMED features to the extracted ECG features
# This keeps all records from `features_df` and adds matching data from `processed_df`
merged_df = pd.merge(features_df, processed_df, on='exam_id', how="inner")

# Display the first few rows of the merged DataFrame and its shape
print("Shape of the merged DataFrame:", merged_df.shape)
merged_df.head()