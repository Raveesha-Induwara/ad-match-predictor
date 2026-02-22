# Ad Match Predictor

A tool that predicts whether two classified ads are related to each other. For example, if someone is **selling a phone** and another person is **looking to buy a phone**, the system figures out that these two ads are a match.

Built for **Sri Lankan classified marketplaces** (like ikman.lk style platforms).

## What Does This Project Do?

Imagine a website where people post ads:
- **Offering ads** — "I'm selling a Samsung Galaxy S21"
- **Wanted ads** — "Looking for a Samsung phone under 100,000 LKR"

This project uses a trained computer model to automatically check if an offering ad matches a wanted ad. It compares things like:
- How similar the words are between the two ads
- Whether they belong to the same category (Electronics, Vehicle, Property)
- How much the titles and descriptions overlap

The model outputs a **Match** or **No Match** prediction along with a confidence percentage.

## Project Structure

```
├── app.py                  → Web app (user interface)
├── train_model.py          → Script that trains the prediction model
├── explore_dataset.py      → Script to explore and sample the dataset
├── requirements.txt        → Python packages needed
├── sampled_dataset_10k.csv → 10,000 ad pairs used for training
├── models/                 → Trained model and related files
└── plots/                  → Charts showing model performance
```

## How to Run

### 1. Install Python packages

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This will:
- Process the dataset
- Train the model
- Save the model to the `models/` folder
- Generate performance charts in the `plots/` folder

### 3. Launch the web app

```bash
streamlit run app.py
```

This opens a browser where you can:
1. Type in an offering ad (title + description + category)
2. Type in a wanted ad (title + description + category)
3. Click **Predict Match** to see the result

## Supported Categories

| Main Category | Subcategories |
|---------------|---------------|
| Electronics | Home Appliances, Mobile Phones, Computers, TVs, Audio, Cameras, etc. |
| Vehicle | Car, Van, Lorry/Truck, Three-wheeler, Bike, Bicycle |
| Property | House, Land, Commercial Property, Apartment, Room & Annex |

## Dataset

- **Source**: Sri Lankan Classified Ads Matching Dataset v1 (54,489 ad pairs)
- **Sampled**: 10,000 pairs used for training
- Each row contains an offering ad and a wanted ad that are known matches
- The training script also creates non-matching pairs so the model can learn the difference

## Tech Stack

- **Python** — programming language
- **LightGBM** — the algorithm that makes predictions (a type of decision-tree model)
- **Streamlit** — creates the web interface
- **SHAP** — explains why the model made a particular prediction
- **scikit-learn** — used for text processing and evaluation metrics
