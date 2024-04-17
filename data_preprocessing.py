import pandas as pd
import numpy as np
import copy
from gensim.models import Word2Vec  # categorical feature to vectors
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re


def ingredient_text_preprocessing(text):
    text = re.sub(r"\d+(\.\d+)?\w*", "", text).strip()
    text = re.sub(r"약간", "", text)
    text = re.sub(r"/", "", text)
    text = re.sub(r"\*", "", text)
    text = re.sub(r"\([^)]*\)", "", text).strip()
    return text


def remove_quantity_and_unit(recipe_ingredients):
    recipe_ingredients = re.sub(r"\[.*?\]", "|", recipe_ingredients)
    recipe_ingredients = recipe_ingredients.replace(" ", "")
    ingredients_list = recipe_ingredients.split("|")

    cleaned_ingredients = [
        ingredient_text_preprocessing(item)
        for item in ingredients_list
        if ingredient_text_preprocessing(item)
    ]

    return cleaned_ingredients


def remove_empty_lists(df):
    df = df[df["CKG_MTRL_CN"].astype(bool)]
    return df


def preprocessing(df, recipe):
    recipe = recipe.dropna(subset=["CKG_NM", "CKG_MTRL_CN", "CKG_INBUN_NM"])
    recipe = recipe.drop_duplicates(subset=["CKG_NM"])
    # recipe = recipe[~recipe['CKG_INBUN_NM'].str.contains("이상")]
    recipe = recipe[
        [
            "CKG_NM",
            "CKG_STA_ACTO_NM",
            "CKG_MTH_ACTO_NM",
            "CKG_MTRL_ACTO_NM",
            "CKG_KND_ACTO_NM",
            "CKG_MTRL_CN",
        ]
    ]

    df = pd.merge(df, recipe, left_on="food", right_on="CKG_NM", how="left")
    df.drop("CKG_NM", axis=1, inplace=True)
    df.drop("food", axis=1, inplace=True)
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    df["CKG_MTRL_CN"] = df["CKG_MTRL_CN"].apply(remove_quantity_and_unit)
    df = remove_empty_lists(df)
    df["CKG_MTRL_CN"] = df["CKG_MTRL_CN"].apply(lambda x: " ".join(x))
    df = df.reset_index(drop=True)
    return df


def recipe_preprocessing(recipe):
    recipe = recipe.dropna(subset=["CKG_NM", "CKG_MTRL_CN", "CKG_INBUN_NM"])
    recipe = recipe.drop_duplicates(subset=["CKG_NM"])
    recipe["CKG_MTRL_CN"] = recipe["CKG_MTRL_CN"].apply(remove_quantity_and_unit)
    recipe = remove_empty_lists(recipe)
    recipe["CKG_MTRL_CN"] = recipe["CKG_MTRL_CN"].apply(lambda x: " ".join(x))

    return recipe


def apply_w2v(sentences, model, num_features):
    def _average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        n_words = 0.0
        for word in words:
            if word in vocabulary:
                n_words = n_words + 1.0
                feature_vector = np.add(feature_vector, model.wv[word])

        if n_words:
            feature_vector = np.divide(feature_vector, n_words)
        return feature_vector

    vocab = set(model.wv.index_to_key)
    feats = [_average_word_vectors(s, model, vocab, num_features) for s in sentences]
    return np.array(feats)


def gen_cat2vec_sentences(data):
    X_w2v = copy.deepcopy(data)
    names = list(X_w2v.columns.values)
    for c in names:
        X_w2v[c] = X_w2v[c].fillna("unknow").astype("category")
        X_w2v[c] = X_w2v[c].cat.rename_categories(
            ["%s %s" % (c, g) for g in X_w2v[c].cat.categories]
        )
    X_w2v = X_w2v.values.tolist()
    return X_w2v


def fit_cat2vec_model(daset, cat_cols, n_cat2vec_feature, n_cat2vec_window):
    X_w2v = gen_cat2vec_sentences(daset.loc[:, cat_cols].sample(frac=0.7))
    for i in X_w2v:
        shuffle(i)
    model = Word2Vec(
        X_w2v, vector_size=n_cat2vec_feature, window=n_cat2vec_window, seed=1
    )
    return model


def label_encoding(series: pd.Series) -> pd.Series:
    my_dict = {}
    series = series.astype(str)
    for idx, value in enumerate(sorted(series.unique())):
        my_dict[value] = idx
    series = series.map(my_dict)

    return series


def data_embedding(train, recipe, member, food):
    member["food"] = food
    data = member[
        [
            "age",
            "gender",
            "height",
            "weight",
            "num_meal",
            "num_exercise",
            "sleep_duration",
            "food",
        ]
    ]

    train = preprocessing(train, recipe)
    train = train.drop(columns="portion")
    test = preprocessing(data, recipe)

    daset = pd.concat([train, test], axis=0)
    cat_cols = train.select_dtypes(include=["object"]).columns

    n_cat2vec_feature = len(cat_cols)
    n_cat2vec_window = len(cat_cols) * 2
    c2v_model = fit_cat2vec_model(daset, cat_cols, n_cat2vec_feature, n_cat2vec_window)

    tr_c2v_matrix = apply_w2v(
        gen_cat2vec_sentences(daset.iloc[: len(train)][cat_cols]),
        c2v_model,
        n_cat2vec_feature,
    )
    te_c2v_matrix = apply_w2v(
        gen_cat2vec_sentences(daset.iloc[len(train) :][cat_cols]),
        c2v_model,
        n_cat2vec_feature,
    )
    tr_c2v_matrix = pd.DataFrame(tr_c2v_matrix)
    te_c2v_matrix = pd.DataFrame(te_c2v_matrix)
    new_columns = [f"cat2vec_{i+1}" for i in range(len(tr_c2v_matrix.columns))]
    tr_c2v_matrix.columns = new_columns
    te_c2v_matrix.columns = new_columns
    train = train.drop(columns=cat_cols)
    test = test.drop(columns=cat_cols)
    train = pd.concat([train, tr_c2v_matrix], axis=1)
    test = pd.concat([test, te_c2v_matrix], axis=1)

    tf_columns = ["CKG_MTRL_CN"]
    tfidf_vectorizer = TfidfVectorizer()
    svd = TruncatedSVD(n_components=7, n_iter=7, random_state=42)

    for col in tf_columns:
        tfidf_matrix = tfidf_vectorizer.fit_transform(daset[col])
        svd_matrix = svd.fit_transform(tfidf_matrix)
        svd_df = pd.DataFrame(svd_matrix)
        new_columns = [f"svd_{col}_{i+1}" for i in range(len(svd_df.columns))]
        svd_df.columns = new_columns
        train = pd.concat([train, svd_df.iloc[: len(train)]], axis=1)

        svd_df_te = svd_df.iloc[len(train) :]
        svd_df_te.index = test.index
        test = pd.concat([test, svd_df_te], axis=1)

    label_columns = [
        "gender",
        "CKG_STA_ACTO_NM",
        "CKG_MTH_ACTO_NM",
        "CKG_MTRL_ACTO_NM",
        "CKG_KND_ACTO_NM",
    ]
    for col in label_columns:
        daset[col] = label_encoding(daset[col])
    train[label_columns] = daset.iloc[: len(train)][label_columns]
    test[label_columns] = daset.iloc[len(train) :][label_columns]

    return test


def extract_servings_ingredients(recipe, menu):
    recipe = recipe.dropna(subset=["CKG_NM", "CKG_MTRL_CN", "CKG_INBUN_NM"])
    recipe = recipe.drop_duplicates(subset=["CKG_NM"])
    menu_data = recipe[recipe["CKG_NM"] == menu]
    servings = int(menu_data["CKG_INBUN_NM"].values[0][0])
    ingredients = menu_data["CKG_MTRL_CN"].values[0].replace("|", "")

    return servings, ingredients
