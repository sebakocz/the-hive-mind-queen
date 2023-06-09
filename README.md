# :bee: The Hive Mind Queen

The Hive Mind Queen is a machine learning-driven project designed to predict the popularity of user-generated cards on Reddit. By gathering and processing card data, including card attributes and ability text, the project aims to uncover patterns and relationships that contribute to a card's success. Utilizing state-of-the-art data preprocessing and natural language processing techniques, The Hive Mind Queen provides valuable insights into the essential factors that make a card resonate with the community, ultimately helping card creators design more appealing and engaging content.

# Table of contents

- [:bee: The Hive Mind Queen](#bee-the-hive-mind-queen)
  - [:mag: Scraping](#mag-scraping)
  - [:world_map: Exploring](#world_map-exploring)
  - [:luggage: Preparing](#luggage-preparing)
  - [:muscle: Training](#muscle-training)
  - [:crystal_ball: Predicting](#crystal_ball-predicting)
 - [:hammer_and_wrench: Installation](#hammer_and_wrench-installation)

## :mag: Scraping
Script for gathering card data and associated Reddit votes. This step involves fetching Reddit votes from a PostgreSQL database and using the card game server's API endpoints to retrieve card attributes. The combined data is then processed and stored in a CSV file.

```bash
py scraper.py
```

### :package: Stored Data

| Column Name | Description                            | Type | Example                  |
| ----------- |----------------------------------------| ---- |--------------------------|
| votes | Number of votes the card has on reddit | int  | 4                        |
| timestamp | Time when the card was posted          | int  | 1580000000               |
| name | Name of the card                       | str  | 'Cat With Frying Pan'    |
| type | Type of the card                       | str  | 'Unit'                   |
| affinity | Affinity of the card                   | str  | 'Neutral'                |
| rarity | Rarity of the card                     | str  | 'Common'                 |
| tribes | Tribes of the card                     | str  | 'Toon Cat'               |
| realm | Realm of the card                      | str  | 'Carachar'               |
| ability_text | Ability text of the card               | str  | 'Summon: Deal 1 damage.' |
| cost | Cost of the card                       | int  | 1                        |
| hp | Health of the card                     | int  | 1                        |
| atk | Attack of the card                     | int  | 1                        |

## :world_map: Exploring
Script for exploring the data. This step involves analyzing the dataset to gain insights and discover patterns. Look out for outliers and missing values.

```bash
jupyter-lab
```
![image](https://user-images.githubusercontent.com/44639352/230596714-9884def8-38ca-48e8-b7ba-1fc261079f7c.png)

<div style="display:flex;">
  <img src="https://user-images.githubusercontent.com/44639352/230597116-db98a878-c333-42c6-92de-ce239ef2f3c2.png" style="width: 48%;">
  <img src="https://user-images.githubusercontent.com/44639352/230596795-1f389ecf-38c7-4249-b5a3-a2b540488be8.png" style="width: 48%;">
</div>

## :luggage: Preparing
Script for preparing the data. This step involves cleaning the data and transforming it into a format that can be used by the model.

```bash
py preparator.py
```

### :broom: Cleaning
The data is cleaned by filling NaN values and removing unnecessary columns.

### :eyeglasses: Binary Classification
The data is converted into a binary classification problem. This is done by creating a new column `popular` and assigning `1` or `0` to the column depending on whether the card has more than 2 votes. A card with more than 2 votes is considered popular.

### :one: One-Hot Encoding
A method to convert categorical variables into binary vectors by creating a separate column for each category. Each row has a 1 in the column corresponding to its category and 0 in all other columns.

Suitable for `type`, `affinity`, and `rarity` columns due to a small number of unique values.

### :hash: Feature Hashing
A technique to transform categorical variables into numerical data by mapping categories to a fixed number of columns using a hashing function. Suitable for high-dimensional data with many categories, as it reduces dimensionality and computational costs but can introduce collisions.

Suitable for `realm` and `tribes` columns due to a large number of unique values.

### :chart_with_downwards_trend: Scaling
Scaling is a process of converting numerical data into a range of values. This is done using the StandardScaler, which standardizes features by removing the mean and scaling to unit variance.

Applied to the `cost`, `hp`, and `atk` columns.

### :speech_balloon: Text Embedding
A method to convert text data into numerical vectors that capture semantic relationships between words or phrases. The script uses the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert the `ability_text` column into numerical vectors.

### :floppy_disk: Saving Preprocessing Components
During the data preparation stage for training, the script saves the hasher, scaler, and TF-IDF vectorizer objects, along with the feature names, to ensure consistency when preparing new data for predictions. These saved components are used to preprocess new data in the same way as the training data, allowing for a seamless integration with the trained model.

The following components are saved:

- `hasher.pkl`: The FeatureHasher object used for hashing `realm` and `tribes` columns.
- `scaler.pkl`: The StandardScaler object used for scaling the `cost`, `hp`, and `atk` columns.
- `tfidf_vectorizer.pkl`: The TfidfVectorizer object used for embedding the `ability_text` column.
- `feature_names.csv`: A CSV file containing the feature names of the training data after preprocessing.

These components are stored in the `train_misc` directory and loaded when preparing new data for predictions.


## :muscle: Training
Script for training the machine learning model. This step involves loading and preprocessing the data, training a RandomForestClassifier, and evaluating its performance.

```bash
py trainer.py
```

### :cake: Split Data
The data is split into a training set (80%) and a testing set (20%).

### :racing_car: Hyperparameter Tuning
A RandomForestClassifier model is instantiated with a random state of 42 for reproducibility. The hyperparameters are tuned using RandomizedSearchCV with cross-validation.

### :heavy_check_mark: Feature Selection
Feature importances are calculated, and the top 10 most important features are selected for the final model.

### :microscope: Model Evaluation
The model is evaluated on the training and testing sets. The evaluation metrics include accuracy, precision, recall, and F1-score. Bootstrap resampling is used to estimate the mean and standard deviation of the test F1-score.

### :man_juggling: Bootstrap Resampling

Bootstrap resampling is used during the model evaluation stage to estimate the mean and standard deviation of the test F1-score. This technique involves creating multiple resampled test datasets by sampling with replacement from the original test set. The model is then evaluated on each of these resampled datasets, and the F1-scores are calculated. The mean and standard deviation of these F1-scores provide an estimate of the model's performance and its variability when applied to new, unseen data. By using bootstrap resampling, we can better understand the model's generalization ability and account for the randomness in the dataset.

## :crystal_ball: Predicting
Script for predicting the popularity of new cards using the trained machine learning model.

```bash
py predictor.py
```

### :factory: Load & Preprocess
Use the `prepare_data()` function from the "Preparation" section with `is_training=False` to preprocess the new card data.

### :outbox_tray: Output
Use the trained model to make predictions for the new card data. Output is printed in the console.

# :hammer_and_wrench: Installation

### Requirements
- Python 3.10
- pip
- JupyterLab

### Setup
1. Clone the repository
2. Install the required packages
```bash
pip install -r requirements.txt
```
3. Copy the `.env.example` file and rename it to `.env`
4. Fill in the required fields in the `.env` file
