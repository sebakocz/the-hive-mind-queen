from concurrent.futures import ThreadPoolExecutor
import joblib
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_data(df, is_training=True):
    # Input validation
    assert "type" in df.columns, "Missing 'type' column in input DataFrame"
    assert "affinity" in df.columns, "Missing 'affinity' column in input DataFrame"
    assert "rarity" in df.columns, "Missing 'rarity' column in input DataFrame"
    assert "tribes" in df.columns, "Missing 'tribes' column in input DataFrame"
    assert "realm" in df.columns, "Missing 'realm' column in input DataFrame"
    assert (
        "ability_text" in df.columns
    ), "Missing 'ability_text' column in input DataFrame"
    assert {"cost", "hp", "atk"}.issubset(
        df.columns
    ), "Missing numerical columns in input DataFrame"
    if is_training:
        assert "votes" in df.columns, "Missing 'votes' column in input DataFrame"
        assert (
            "timestamp" in df.columns
        ), "Missing 'timestamp' column in input DataFrame"

    # Binary Classification
    if is_training:
        df["popular"] = df["votes"].apply(lambda x: 1 if x > 2 else 0)
        # count popular cards
        print(df["popular"].value_counts())

    # Fill NaN values
    df["tribes"].fillna("-", inplace=True)
    df["realm"].fillna("-", inplace=True)
    df["affinity"].fillna("Neutral", inplace=True)
    df["ability_text"].fillna("", inplace=True)
    df.fillna(0, inplace=True)

    # One-Hot Encoding
    df = pd.concat([df, pd.get_dummies(df["type"], prefix="type")], axis=1)
    df = pd.concat([df, pd.get_dummies(df["affinity"], prefix="affinity")], axis=1)
    df = pd.concat([df, pd.get_dummies(df["rarity"], prefix="rarity")], axis=1)
    df.drop(["type", "affinity", "rarity"], axis=1, inplace=True)

    # Feature Hashing
    df["tribes"] = df["tribes"].apply(lambda x: x.split())
    df["realm"] = df["realm"].apply(lambda x: [x])

    if is_training:
        hasher = FeatureHasher(n_features=20, input_type="string")
        hasher.fit(df["tribes"])
        hasher.fit(df["realm"])
        joblib.dump(hasher, "../../data/train_misc/hasher.pkl")
    else:
        hasher = joblib.load("../../data/train_misc/hasher.pkl")

    tribes = hasher.transform(df["tribes"])
    realms = hasher.transform(df["realm"])
    tribes = pd.DataFrame(tribes.toarray()).add_prefix("tribe_")
    realms = pd.DataFrame(realms.toarray()).add_prefix("realm_")

    df = pd.concat([df, tribes, realms], axis=1)
    df.drop(["tribes", "realm"], axis=1, inplace=True)

    # Scaling numerical columns
    num_cols = ["cost", "hp", "atk"]
    if is_training:
        scaler = StandardScaler()
        scaler.fit(df[num_cols])
        joblib.dump(scaler, "../../data/train_misc/scaler.pkl")
    else:
        scaler = joblib.load("../../data/train_misc/scaler.pkl")

    df[num_cols] = scaler.transform(df[num_cols])

    # TF-IDF for ability_text
    if is_training:
        tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        tfidf_matrix = tfidf_vectorizer.fit_transform(df["ability_text"])
        joblib.dump(tfidf_vectorizer, "../../data/train_misc/tfidf_vectorizer.pkl")
    else:
        tfidf_vectorizer = joblib.load("../../data/train_misc/tfidf_vectorizer.pkl")
        tfidf_matrix = tfidf_vectorizer.transform(df["ability_text"])

    ability_tfidf_df = pd.DataFrame(tfidf_matrix.toarray()).add_prefix("ability_tfidf_")
    df = pd.concat([df, ability_tfidf_df], axis=1)
    df.drop(["ability_text"], axis=1, inplace=True)

    # Ensure new cards data has the same columns as the training data
    if not is_training:
        # Load training data feature names
        train_feature_names = (
            pd.read_csv("../../data/train_misc/feature_names.csv", header=None)
            .squeeze()
            .tolist()
        )

        # Add missing columns in new cards data with zeros
        missing_columns = set(train_feature_names) - set(df.columns)
        for col in missing_columns:
            df[col] = 0

        # Make sure the new cards data has the same column order as the training data
        df = df[train_feature_names]

    # Remove unnecessary columns
    if is_training:
        df.drop("timestamp", axis=1, inplace=True)
        df.drop("votes", axis=1, inplace=True)
        df.drop("name", axis=1, inplace=True)

    # Save feature names
    if is_training:
        df.columns.to_series().to_csv(
            "../../data/train_misc/feature_names.csv", index=False
        )

    return df


def main():
    # Load data
    data = pd.read_csv("../../data/processed_data/cards_cleaned.csv")

    # Process data
    data = prepare_data(data)

    # Save data
    data.to_csv("../../data/processed_data/cards_prepared.csv", index=False)


if __name__ == "__main__":
    main()
