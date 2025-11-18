```markdown
# SMS Spam Classifier Project

This project implements a **SMS Spam Classification system** using three models: **Naive Bayes (NB), Neural Network (NN), and SVM**. The system can classify an SMS as **Spam** or **Ham** and compare the performance of all three models.

---

## Project Structure

```

**pycache**/
app.py
data/
model_manual.py
model_manual_nb.py
model_manual_svm.py
models/
requirements.txt
scripts/
README.md

````

### Important Directories and Files

- **app.py**  
  Streamlit application for SMS classification and model comparison.

- **data/SMSSpamCollection**  
  Dataset containing labeled SMS messages (`spam` or `ham`).

- **models/**  
  Pre-trained models and vectorizers:
  - `models/nb/` → Naive Bayes model (`manual_nb_model.joblib`)
  - `models/nn/` → Neural Network weights (`nn_weights.npz`) and TF-IDF vectorizer
  - `models/svm/` → SVM model files and TF-IDF vectorizer

- **scripts/**  
  Scripts used for training and preprocessing:
  - `train_nb.py` → Train the Naive Bayes model and save it in `models/nb/`
  - `train_nn.py` → Train the Neural Network model and save weights in `models/nn/`
  - `train_svm.py` → Train the SVM model and save it in `models/svm/`
  - `preprocessing.py` → Functions to clean and preprocess the SMS text data before training

- **model_manual.py** → Contains the **ManualNN** class for neural network implementation  
- **model_manual_nb.py** → Contains the **ManualNB** class for Naive Bayes implementation  
- **model_manual_svm.py** → Contains the **ManualSVM** class for SVM implementation  

---

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> Make sure you are using Python 3.10 or later.

---

## Running the Project

Run the Streamlit application:

```bash
streamlit run app.py
```

This will open the app in your default browser.

**Features:**

1. **SMS Prediction:**

   * Select a model (Naive Bayes, Neural Network, or SVM)
   * Type an SMS message
   * Get classification: **Spam** (red) or **Ham** (green)

2. **Model Comparison:**

   * Displays metrics for all three models: Accuracy, Precision, Recall, F1-score, ROC-AUC
   * Confusion matrices for each model
   * Some misclassified examples

---

## Notes

* Training scripts are provided under `scripts/` but the README focuses on running the final Streamlit app.
* All models are pre-trained and saved in the `models/` folder.

---

## Authors

* Bhargavi Mahajan
* Arya Jha
* Ria Kumari
* Shreya Srivastava
* Suhani Verma

---

## License

This project is licensed under the MIT License.

```

---

I can also make a **more visual README** with tables for scripts, models, and sample predictions if you want it to look professional for GitHub.  

Do you want me to do that?
```
