# NYC Taxi Travel Time Prediction

This project serves as a training task for MLOps, aiming to predict the travel time of New York City (NYC) taxis based on various features. The goal is to estimate the duration of a taxi trip using machine learning models and relevant data, allowing for more accurate planning and prediction of travel times.

## Features

The application utilizes several features to predict taxi travel time:

1. **Datetime Features:**
   - Date
   - Hour
   - Day of the week

2. **Holiday Features:**
   - Binary feature indicating whether it is a holiday or not.

3. **OSRM Features:**
   - Total distance
   - Total travel time
   - Number of steps

4. **Geographical Features:**
   - Haversine distance
   - Direction

5. **Cluster Features:**
   - Geographic clustering based on start and end points of the trip.

6. **Weather Features:**
   - Temperature
   - Visibility
   - Wind speed
   - Precipitation
   - Events (e.g., rain, snow)

## Usage

1. **Data Preparation:**
   - Ensure you have the necessary data files for training and testing the model.
   - Data includes taxi trip details, holiday information, OSRM data, and weather features.

2. **DVC Pipeline:**
   - The project uses DVC (Data Version Control) for managing and versioning datasets.
   - Run `dvc pull` to fetch the necessary datasets.
   - Run `dvc repro` to reproduce the pipeline.

3. **Feature Engineering:**
   - Run the provided scripts to add datetime, holiday, OSRM, geographical, cluster, and weather features to the dataset.

4. **Model Training:**
   - Train a machine learning model using the provided scripts. Hyperparameters can be configured in the `params.yaml` file.

5. **Evaluation:**
   - Evaluate the model's performance using the evaluation script, providing the trained model and test data.

6. **Prediction:**
   - Use the trained model to predict travel times for new taxi trips.

## MLOps Training Task

- This project is designed as a training task for MLOps practices.
- It incorporates DVC for data version control and follows a structured pipeline for model development and deployment.

## Scripts

- `add_features.py`: Adds various features to the dataset.
- `best_features.py`: Selects the best features for model training using the SelectKBest method.
- `ohe.py`: Performs one-hot encoding on categorical features.
- `split_data.py`: Splits the dataset into training and testing sets.
- `model_train.py`: Trains a Gradient Boosting Regressor model.
- `evaluate.py`: Evaluates the trained model using test data.

## Requirements

- Python 3
- NumPy
- pandas
- scikit-learn
- DVC

## Notes

- The application assumes proper data file formats and directory structures. 

- Ensure you have the required Python packages installed. You can install them using:

  ```bash
  pip install -r requirements.txt
