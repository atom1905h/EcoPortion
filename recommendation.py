import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def remove_quantity_and_unit(recipe_ingredients):
    recipe_ingredients = recipe_ingredients.replace("[재료]", "")
    recipe_ingredients = recipe_ingredients.replace("[양념]", "|")
    ingredients_list = recipe_ingredients.split("|")

    cleaned_ingredients = []
    for ingredient in ingredients_list:
        parts = ingredient.strip().split(" ")
        ingredient_name = " ".join(parts[:-1])
        cleaned_ingredient = re.sub(r"\[.*?\]", "", ingredient_name).strip()
        if cleaned_ingredient:
            cleaned_ingredients.append(cleaned_ingredient)

    return cleaned_ingredients


def remove_empty_lists(df):
    df = df[df["CKG_MTRL_CN"].astype(bool)]
    return df


def recommend_similar_food(food_name, top_n=5):
    sub_recipe = pd.read_csv("sub_recipe.csv")
    sub_recipe["CKG_MTRL_CN"] = sub_recipe["CKG_MTRL_CN"].apply(
        remove_quantity_and_unit
    )
    sub_recipe = remove_empty_lists(sub_recipe)
    sub_recipe["CKG_MTRL_CN"] = sub_recipe["CKG_MTRL_CN"].apply(lambda x: " ".join(x))
    recipe = pd.read_csv("recipe.csv")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sub_recipe["CKG_MTRL_CN"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    similarity_matrix = pd.DataFrame(similarity_matrix)
    similarity_matrix.index = sub_recipe["CKG_NM"]

    try:
        similar_indices = (
            similarity_matrix.loc[food_name].sort_values(ascending=False).index[:top_n]
        )
        similar_food = [similarity_matrix.iloc[i].name for i in similar_indices]
        return similar_food
    except KeyError:
        pop_list = recipe["INQ_CNT"].sort_values(ascending=False).index[:5].to_list()
        pop_food = recipe.loc[pop_list, "CKG_NM"].to_list()
        return pop_food
