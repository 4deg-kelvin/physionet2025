1. Using combined_prna outputs plus features found in EDA folder, create feature_embeddings for each datapoint in STRONG dataset (PTB, samitrop)
2. Train XGBoost model on just STRONG dataset feature_embeddings
3. Forward pass XGBoost model (not train) on WEAK dataset (code15) to get predicted probabilities for each datapoint. Also save the feature_embeddings for the WEAK dataset as well
4. Pass predicted probabilities AND feature embeddings for WEAK dataset into cleanlabs, then use this code from CleanLabs


# cleanlab works with **any classifier**. Yup, you can use PyTorch/TensorFlow/OpenAI/XGBoost/etc.
cl = cleanlab.classification.CleanLearning(sklearn.YourFavoriteClassifier())

# cleanlab finds data and label issues in **any dataset**... in ONE line of code!
label_issues = cl.find_label_issues(data, labels)

# cleanlab trains a robust version of your model that works more reliably with noisy data.
cl.fit(data, labels)

# cleanlab estimates the predictions you would have gotten if you had trained with *no* label issues.
cl.predict(test_data)

# A universal data-centric AI tool, cleanlab quantifies class-level issues and overall data quality, for any dataset.
cleanlab.dataset.health_summary(labels, confident_joint=cl.confident_joint)


(credit Kelvin for instructions)
