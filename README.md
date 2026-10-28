# HomeValue — House Price Estimator

A Streamlit machine-learning demo that estimates a property's price from five features: bedrooms, bathrooms, living area, house condition, and nearby schools.

## Demo

The app loads the included scikit-learn model and presents estimates in the original dataset's price units. The training data does not establish a reliable currency or market context, so the app intentionally does not label the estimate with a currency symbol. This is an educational project, not a property appraisal.

## Run locally

Use Python 3.11 (or another version supported by the pinned dependencies).

```bash
python -m venv .venv
```

Activate the environment, then install and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this folder (`app.py`, `model.pkl`, `requirements.txt`, `README.md`, and `.gitignore`).
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and authorize the repository.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Deploy. Streamlit installs the packages listed in `requirements.txt` and loads `model.pkl` from the repository root.

No secrets or external API keys are required. Do not commit credentials if you extend the project later.

## Project structure

```text
.
├── app.py             # Streamlit user interface and prediction flow
├── model.pkl          # Trained GridSearchCV estimator
├── requirements.txt   # Pinned runtime dependencies
├── README.md
└── .gitignore
```

## Model inputs

| Feature | Dataset field | Input range |
| --- | --- | ---: |
| Bedrooms | `number of bedrooms` | 1–33 |
| Bathrooms | `number of bathrooms` | 0.5–8 |
| Living area | `living area` | 370–13,540 sq ft |
| Condition | `condition of the house` | 1–5 |
| Nearby schools | `Number of schools nearby` | 1–3 |

The feature names and order match the estimator's training schema. The app sends a named pandas DataFrame to the model.

## Resume-ready summary

**House Price Estimator | Python, scikit-learn, Streamlit** — Built an interactive regression app that serves a trained GridSearchCV model, validates property inputs, and exposes estimates through a responsive web interface.

Only say “deployed” on your resume after you have published the app and confirmed its public URL works. You can then add the live demo link and your GitHub repository link.
