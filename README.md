# Credit Card Fraud Detection -- MLOps Pipeline

An end-to-end machine learning operations (MLOps) pipeline for detecting fraudulent credit card transactions. The system trains and serves multiple classification models behind a Flask web interface and uses LLM-powered natural language explanations to make predictions interpretable.

---

## Features

- **Multi-model inference** -- choose between Logistic Regression, Random Forest, and XGBoost at prediction time.
- **SMOTE-based class balancing** -- handles the extreme class imbalance inherent in fraud datasets using Synthetic Minority Over-sampling.
- **Configurable cross-validation** -- supports KFold, StratifiedKFold, and Leave-One-Out validation strategies.
- **LLM-powered explainability** -- generates natural language explanations for each prediction via OpenAI (GPT-3.5-Turbo) and LangChain.
- **Experiment tracking** -- logs parameters, metrics (accuracy, precision, recall, F1), ROC curves, and model artifacts to MLflow.
- **Data version control** -- tracks large data files with DVC so the dataset stays reproducible without bloating the repository.
- **Interactive web dashboard** -- a Flask-based UI where users enter transaction features, select a model, and receive a fraud/not-fraud prediction with probability and explanation.
- **Docker-ready** -- ships with a production Dockerfile targeting Google Cloud Run.
- **Feature scaling** -- StandardScaler applied to Time and Amount features; scaler persisted alongside model artifacts.

---

## Tech Stack

| Layer              | Technology                                              |
| ------------------ | ------------------------------------------------------- |
| Language           | Python 3.12                                             |
| Web Framework      | Flask                                                   |
| ML Models          | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Class Balancing    | imbalanced-learn (SMOTE)                                |
| Explainability     | LangChain + OpenAI GPT-3.5-Turbo                       |
| Experiment Tracking| MLflow                                                  |
| Data Versioning    | DVC                                                     |
| Containerization   | Docker                                                  |
| Deployment Target  | Google Cloud Run                                        |
| Monitoring (deps)  | Prometheus Client, Grafana API                          |
| Testing            | pytest                                                  |

---

## Architecture / How It Works

```
                         +------------------+
                         |   Raw Dataset    |
                         | (creditcard.csv) |
                         +--------+---------+
                                  |
                          DVC-tracked data
                                  |
                    +-------------v--------------+
                    |   Data Ingestion & Prep    |
                    |  src/data_ingestion.py     |
                    |  - Load CSV                |
                    |  - Scale Time & Amount     |
                    |  - Save StandardScaler     |
                    +-------------+--------------+
                                  |
                    +-------------v--------------+
                    |   Model Training           |
                    |  src/train.py              |
                    |  - SMOTE oversampling      |
                    |  - Cross-validation        |
                    |  - Train LR / RF / XGB     |
                    |  - Log to MLflow           |
                    |  - Save .pkl artifacts     |
                    |  - Plot ROC curves         |
                    +-------------+--------------+
                                  |
                    +-------------v--------------+
                    |   Flask Web Application    |
                    |  app.py                    |
                    |  - Load saved models       |
                    |  - Accept user input       |
                    |  - Return prediction +     |
                    |    probability + LLM       |
                    |    explanation              |
                    +----------------------------+
```

1. **Data Ingestion** -- `src/data_ingestion.py` loads the raw Kaggle credit card dataset, scales the `Time` and `Amount` columns with `StandardScaler`, drops the originals, and persists the scaler as `models/scaler.pkl`.
2. **Training** -- `src/train.py` splits data 80/20, applies SMOTE to the training set, then trains Logistic Regression, Random Forest, and XGBoost. Each run is logged to MLflow with accuracy, precision, recall, F1-score, and a ROC curve artifact. Serialized models are saved under `models/`.
3. **Serving** -- `app.py` loads the three pre-trained models and scaler at startup. Users submit transaction features through the web form, choose a model, and receive a fraud/not-fraud prediction with probability. An LLM explanation is generated on the fly via LangChain.

---

## Project Structure

```
creditcardfrauddetection-mlops/
|-- app.py                          # Flask application (prediction + LLM explanation)
|-- Dockerfile                      # Production container image (Python 3.12-slim)
|-- requirements.txt                # Python dependencies
|-- .dockerignore                   # Files excluded from Docker build context
|-- .dvc/                           # DVC configuration
|   |-- config
|   +-- .gitignore
|-- .dvcignore
|-- .gitattributes
|-- .gitignore
|-- data/
|   +-- raw/
|       |-- creditcard.csv.dvc      # DVC pointer to the raw dataset
|       +-- .gitignore
|-- models/
|   |-- LogisticRegression_model.pkl
|   |-- RandomForest_model.pkl
|   |-- XGBoost_model.pkl
|   |-- scaler.pkl
|   |-- LogisticRegression_roc_curve.png
|   |-- RandomForest_roc_curve.png
|   +-- XGBoost_roc_curve.png
|-- notebooks/
|   |-- data_analysis.ipynb
|   +-- fraud_detection_modeling.ipynb
|-- src/
|   |-- data_ingestion.py           # Data loading and preprocessing
|   +-- train.py                    # Model training, evaluation, and MLflow logging
|-- static/
|   +-- css/
|       +-- style.css               # Dashboard styling
|-- templates/
|   +-- index.html                  # Fraud Detection Dashboard UI
+-- README.md
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- pip
- (Optional) Docker
- (Optional) DVC, if you need to pull the raw dataset
- An OpenAI API key (required for LLM-based prediction explanations)

### Installation

```bash
# Clone the repository
git clone https://github.com/akshay1389/creditcardfrauddetection-mlops.git
cd creditcardfrauddetection-mlops

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Pulling the Dataset (DVC)

If the raw dataset is stored in a DVC remote, pull it with:

```bash
dvc pull
```

This will download `data/raw/creditcard.csv`.

### Training the Models

```bash
python src/train.py
```

This will:
- Load and preprocess the dataset.
- Apply SMOTE to balance classes.
- Train Logistic Regression, Random Forest, and XGBoost.
- Log all runs to MLflow (viewable at `http://localhost:5000` via `mlflow ui`).
- Save model `.pkl` files and ROC curve plots to `models/`.

### Running the Application

```bash
python app.py
```

The Flask server starts on `http://0.0.0.0:8080`. Open a browser and navigate to `http://localhost:8080` to access the Fraud Detection Dashboard.

---

## Docker

### Build the Image

```bash
docker build -t credit-card-fraud-detection .
```

### Run the Container

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=your_openai_api_key_here \
  credit-card-fraud-detection
```

The application will be available at `http://localhost:8080`.

### Deploy to Google Cloud Run

The Dockerfile is configured to expose port 8080, which is the default expected by Google Cloud Run:

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/credit-card-fraud-detection
gcloud run deploy credit-card-fraud-detection \
  --image gcr.io/YOUR_PROJECT_ID/credit-card-fraud-detection \
  --platform managed \
  --set-env-vars OPENAI_API_KEY=your_openai_api_key_here \
  --allow-unauthenticated
```

---

## Screenshots

> Screenshots of the Fraud Detection Dashboard will be added here.

| View                     | Screenshot               |
| ------------------------ | ------------------------ |
| Dashboard (input form)   | _placeholder_            |
| Prediction result        | _placeholder_            |
| MLflow experiment runs   | _placeholder_            |

---

## Future Enhancements

- **CI/CD pipeline** -- Add GitHub Actions workflows for automated testing, linting, and container builds on every push.
- **Model monitoring** -- Integrate Prometheus and Grafana dashboards (dependencies already included) to track prediction latency, throughput, and data drift in production.
- **Real-time streaming** -- Ingest transactions from Apache Kafka or Pub/Sub for low-latency, event-driven fraud scoring.
- **Additional models** -- Incorporate deep learning approaches (e.g., autoencoders, LSTM) using the TensorFlow dependency already in the stack.
- **Feature store** -- Centralize feature engineering with a feature store (e.g., Feast) to share features across training and serving.
- **A/B testing** -- Serve multiple model versions simultaneously and route traffic to compare performance in production.
- **Hyperparameter tuning** -- Add Optuna or Ray Tune integration for automated hyperparameter optimization with MLflow tracking.
- **UMAP visualization** -- Leverage the umap-learn dependency to build interactive dimensionality-reduction plots of fraud vs. legitimate clusters.
- **Authentication and RBAC** -- Secure the web dashboard with user authentication and role-based access control for enterprise deployments.
- **Batch prediction mode** -- Support CSV upload for bulk transaction scoring in addition to the single-transaction form.

---

## License

This project is provided as-is for educational and research purposes. See the repository for any license information.
