import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer


def one_hot_encode(df, feature_name, categories):
    for category in categories:
        df['{}_{}'.format(feature_name, category)] = (df[feature_name] == category) * 1
    del df[feature_name]


def preprocess(df):
    """This function takes a dataframe and preprocesses it so it is
    ready for the training stage.

    The DataFrame contains columns used for training (features)
    as well as the target column.

    :param df: the dataset
    :type df: pd.DataFrame
    :return: X, y, X_eval
    """

    features_for_model_0_1 = ['goal', 'creator', 'deadline', 'created_at', 'country', 'launched_at', 'static_usd_rate',
                              'category', 'profile', 'state', "evaluation_set"]

    numeric_df = df[features_for_model_0_1]
    numeric_df["goal_in_usd"] = numeric_df["goal"] * numeric_df["static_usd_rate"]
    country_categories = numeric_df["country"].unique()
    one_hot_encode(numeric_df, "country", country_categories)
    numeric_df["campaign_duration_ms"] = numeric_df['deadline'] - numeric_df['launched_at']
    numeric_df["campaign_prep_ms"] = numeric_df['launched_at'] - numeric_df['created_at']
    numeric_df["campaign_life_ms"] = numeric_df['deadline'] - numeric_df['created_at']
    numeric_df["category_id"] = numeric_df.category.apply(lambda x: json.loads(x)['id'])
    cat_categories = numeric_df["category_id"].unique()
    one_hot_encode(numeric_df, "category_id", cat_categories)
    numeric_df["profile_state"] = numeric_df.profile.apply(lambda x: json.loads(x)["state"])
    numeric_df["profile_state"] = (numeric_df["profile_state"] == "active") * 1
    numeric_df["profile_state_changed_at"] = numeric_df.profile.apply(lambda x: json.loads(x)["state_changed_at"])
    numeric_df["tmp1"] = numeric_df['deadline'] - numeric_df["profile_state_changed_at"]
    numeric_df["tmp2"] = numeric_df['launched_at'] - numeric_df["profile_state_changed_at"]

    # Calculate if a person has more than 2 success
    numeric_df["id_person"] = numeric_df.creator.apply(lambda x: json.loads(x)['id'])
    df_mask = numeric_df.groupby("id_person").sum()
    msk_id = df_mask.state
    array_id = df_mask[msk_id > 2].index
    numeric_df["id_person"] = (numeric_df["id_person"].isin(array_id)) * 1

    del numeric_df['goal']
    del numeric_df['static_usd_rate']
    del numeric_df['deadline']
    del numeric_df['launched_at']
    del numeric_df['created_at']
    del numeric_df['category']
    del numeric_df['profile']
    del numeric_df['creator']
    del numeric_df["profile_state_changed_at"]

    # Scale the non binary features
    numeric_df[["goal_in_usd", "campaign_life_ms", "campaign_prep_ms", "campaign_duration_ms", "tmp1",
                "tmp2"]] = PowerTransformer().fit_transform(numeric_df[["goal_in_usd", "campaign_life_ms",
                                                                        "campaign_prep_ms",
                                                                        "campaign_duration_ms", "tmp1", "tmp2"]])

    msk_eval = numeric_df.evaluation_set
    numeric_df = numeric_df.drop(["evaluation_set"], axis=1)
    X = numeric_df[msk_eval == 0].drop(["state"], axis=1)
    y = numeric_df[msk_eval == 0]["state"]
    X_eval = numeric_df[msk_eval == 1].drop(["state"], axis=1)
    return X, y, X_eval


def train(X, y):
    """Trains a new model on X and y and returns it.

    :param X: your processed training data
    :type X: pd.DataFrame
    :param y: your processed label y
    :type y: pd.DataFrame with one column or pd.Series
    :return: a trained model
    """

    model = LogisticRegression(C=0.4)
    # model = DecisionTreeClassifier(max_depth=7)
    model.fit(X, y)
    return model


def predict(model, X_test):
    """This functions takes your trained model as well
    as a processed test dataset and returns predictions.

    :param model: your trained model
    :param X_test: a processed test set
    :return: y_pred, your predictions
    """

    y_pred = model.predict(X_test)

    return y_pred
