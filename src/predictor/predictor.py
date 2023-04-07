import joblib
import pandas as pd

from src.preparator.preparator import prepare_data


def main():
    new_card_data = pd.read_csv("../../data/input_data/new_card_data.csv")

    # Preprocess the new card data using the prepare_data() function with is_training=False
    new_card_data_preprocessed = prepare_data(new_card_data, is_training=False)

    # Load the selected features from the training script
    with open("../../data/train_misc/selected_features.txt", "r") as f:
        selected_features = [line.strip() for line in f]

    # Subset the preprocessed new card data using the selected features
    new_card_data_preprocessed_selected = new_card_data_preprocessed[selected_features]

    # Load the trained model from the saved file
    model = joblib.load("../../models/hive_mind_queen_model.pkl")

    # Use the trained model to make predictions for the new card data
    predictions = model.predict(new_card_data_preprocessed_selected)

    # Print the predicted popularity for each new card
    print(predictions)


if __name__ == "__main__":
    main()
