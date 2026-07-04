# Reproducibility Checklist

- [x] Fixed random seed (SEED = 42) is used for reproducible results.
- [x] Dataset is split into 70% training, 10% validation and 20% testing using stratified sampling.
- [x] Text preprocessing steps are explained in the README and implemented in train.py.
- [x] The `subject` and `date` columns are removed to avoid data leakage.
- [x] TF-IDF is fitted only on the training data to avoid the data leakage.
- [x] Required Python libraries are listed in requirements.txt.
- [x] The project can be executed using a single command: `python train.py`.
- [x] Running the project again with the same dataset and environment produces the same evaluation metrics because the random seed is fixed.
- [x] The model is evaluated using Accuracy, Precision, Recall, F1-score and AUC-ROC.
- [x] A confusion matrix image is generated and saved in the results folder.
- [x] The choice of model, assumptions and limitations are explained in the README.