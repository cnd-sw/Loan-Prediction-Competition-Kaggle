import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df['person_age'].fillna(df['person_age'].median(), inplace=True)
    df['person_income'].fillna(df['person_income'].median(), inplace=True)
    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
    df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
    df['cb_person_cred_hist_length'].fillna(df['cb_person_cred_hist_length'].median(), inplace=True)
    df['cb_person_default_on_file'].fillna(df['cb_person_default_on_file'].mode()[0], inplace=True)
    df['person_home_ownership'].fillna(df['person_home_ownership'].mode()[0], inplace=True)
    df['loan_intent'].fillna(df['loan_intent'].mode()[0], inplace=True)
    df['loan_grade'].fillna(df['loan_grade'].mode()[0], inplace=True)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = fill_missing_values(train_df)
    test_df = fill_missing_values(test_df)

    train_df = encode_categorical(train_df)
    test_df = encode_categorical(test_df)

    if 'id' in train_df.columns:
        train_df.drop('id', axis=1, inplace=True)
    if 'id' in test_df.columns:
        test_df.drop('id', axis=1, inplace=True)

    return train_df, test_df


if __name__ == "__main__":
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train, test = load_data(train_path, test_path)
    train, test = preprocess_data(train, test)

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print(train.head(2))