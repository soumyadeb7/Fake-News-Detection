# Fake News Detection using TF-IDF and Logistic Regression

**Author:** Soumyadeb Bakshi

## About the Project

This project is a binary text classification system that predicts whether a news article is **Fake** or **Real**.

For this project, I used **TF-IDF** to convert the news articles into numerical features and **Logistic Regression** as the classification model.

I selected this approach because it is simple, fast, easy to explain, and performs well on text classification problems.

---

## Project Structure

```
Soumyadeb_Bakshi/

│── train.py
│── README.md
│── requirements.txt
│── reproducibility_checklist.md

├── data/
│     ├── Fake.csv
│     └── True.csv

└── results/
      ├── metrics.json
      └── confusion_matrix.png
```

---

## Dataset

The project uses two CSV files.

- Fake.csv
- True.csv

These files are already included inside the **data** folder.

---

## Libraries Used

The following Python libraries are required.

- pandas
- numpy
- scikit-learn
- matplotlib

Install them using

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open the project folder in the terminal and run

```bash
python train.py
```

The program will automatically

- load the dataset
- preprocess the text
- split the data
- train the model
- evaluate the model
- save the results inside the **results** folder

---

## Data Preprocessing

Before training the model, I cleaned the text by

- converting everything to lowercase
- removing URLs
- removing numbers and special characters
- removing extra spaces
- combining the title and article text into one column

These steps help keep the input text consistent.

---

## Feature Extraction

Machine learning models cannot understand raw text.

So I used **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert every news article into numerical values.

TF-IDF also reduces the importance of very common words while giving higher importance to more informative words.

---

## Model Used

I used **Logistic Regression** for classification.

I selected this model because

- it is simple to understand
- it trains quickly
- it works well with TF-IDF features
- it gives probability scores that can be used to calculate AUC-ROC
- it is easy to interpret

---

## Train, Validation and Test Split

The dataset is divided into

- Training : 70%
- Validation : 10%
- Testing : 20%

A stratified split is used so that both fake and real news are distributed evenly across all three datasets.

---

## Evaluation Metrics

The model is evaluated using

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

A confusion matrix is also generated to understand where the model makes mistakes.

---

## Output

After running the program, the following files are created inside the **results** folder.

```
metrics.json
confusion_matrix.png
```

---

## Reproducibility

To make the results reproducible, I fixed the random seed (`SEED = 42`).

This ensures that the dataset split and model results remain the same every time the project is run using the same dataset and environment.

---

## Assumptions

- Fake and real news contain different writing patterns.
- TF-IDF can capture useful information from the text.
- Logistic Regression can learn these patterns from the training data.

---

## Limitations

- The model only learns from the dataset provided.
- It cannot verify whether a news article is factually true.
- Performance depends on the quality of the dataset.
- TF-IDF does not understand the actual meaning or context of words.

---

## Conclusion

This project demonstrates a complete machine learning pipeline for fake news detection, starting from text preprocessing and feature extraction to model training and evaluation. The implementation is fully reproducible and follows the required train, validation and test split mentioned in the assignment.