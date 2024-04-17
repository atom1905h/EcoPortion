import streamlit as st
import csv
from recommendation import recommend_similar_food
import pandas as pd
from data_preprocessing import (
    data_embedding,
    recipe_preprocessing,
    extract_servings_ingredients,
)
from load_predict import load_models, predict_test_data
from chat_bot import Chatbot
import numpy as np
from identify_missing_ingredients import (
    check_ingredient_status,
    identify_necessary_ingredients,
)
from ultralytics import YOLO
from recipe_crawling import recipe_crawling


@st.cache_data
def get_data():
    train = pd.read_csv("train.csv")
    recipe = pd.read_csv("recipe.csv")
    pro_recipe = recipe_preprocessing(recipe)
    model = YOLO("models/best.pt")
    return train, recipe, pro_recipe, model


train, recipe, pro_recipe, model = get_data()


def add_newlines(text):
    new_text = ""
    for char in text:
        if char == "[":
            new_text += "\n"
            new_text += char
        elif char == "]":
            new_text += char
            new_text += "\n"
        else:
            new_text += char
    return new_text


def save_member(
    name, age, gender, height, weight, num_meal, num_exercise, num_sleep, food1, food2
):
    fieldnames = [
        "name",
        "age",
        "gender",
        "height",
        "weight",
        "num_meal",
        "num_exercise",
        "sleep_duration",
        "food1",
        "food2",
    ]
    with open("members.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {
                "name": name,
                "age": age,
                "gender": gender,
                "height": height,
                "weight": weight,
                "num_meal": num_meal,
                "num_exercise": num_exercise,
                "sleep_duration": num_sleep,
                "food1": food1,
                "food2": food2,
            }
        )
    st.success("데이터가 성공적으로 저장되었습니다.")


def register_member():
    st.title("회원 가입")
    name = st.text_input("이름")
    age = st.number_input("나이", min_value=0, max_value=150, step=1)
    gender = st.selectbox("성별", ["남성", "여성"])
    height = st.number_input("키", min_value=0, max_value=300, step=1)
    weight = st.number_input("체중", min_value=0, max_value=200, step=1)
    num_meal = st.number_input("하루 식사 횟수", min_value=0, max_value=10, step=1)
    num_exercise = st.number_input("일주일간 평균 운동 횟수", min_value=0, max_value=20, step=1)
    num_sleep = st.number_input("하루 평균 수면 시간", min_value=0, max_value=24, step=1)
    food1 = st.text_input("1번 선호하는 음식")
    food2 = st.text_input("2번 선호하는 음식")

    if st.button("회원 가입"):
        save_member(
            name,
            age,
            gender,
            height,
            weight,
            num_meal,
            num_exercise,
            num_sleep,
            food1,
            food2,
        )


@st.cache_data
def food_recommendation(member):
    food1 = member["food1"].values[0]
    reco_list = recommend_similar_food(food1)
    for item in reco_list:
        st.write("- " + item)


def predict(member):
    menu = st.text_input("메뉴 입력")
    if menu:
        if menu in recipe["CKG_NM"].unique():
            test = data_embedding(train, recipe, member, menu)
            lgb_models = load_models(name="lgb")
            xgb_models = load_models(name="xgb")
            cat_models = load_models(name="cat")
            test_predictions = (
                0.2 * predict_test_data(lgb_models, test)
                + 0.4 * predict_test_data(xgb_models, test)
                + 0.4 * predict_test_data(cat_models, test)
            )
            test_predictions = np.round(np.sum(test_predictions), 1)

            servings, ingredients = extract_servings_ingredients(recipe, menu)
            question_1 = f"{servings}인분 재료가 {ingredients} 일 때 {test_predictions}인분의 재료의 양은? 각 재료의 양에 {servings}을 나누고 {test_predictions}을 곱해줘."
            formatted_sentences = recipe_crawling(pro_recipe, menu)
            question_2 = f"{servings}인분일 때 조리 순서는 {formatted_sentences}인데 {test_predictions}인분에 맞춰서 기존 조리 순서에서 재료의 양만 바꿔줘. 답변시작은 [조리 순서]로 해줘"
            chatbot = Chatbot()
            st.subheader("재료")
            st.write(add_newlines(chatbot.get_send_msg(question_1)))
            st.subheader("조리 순서")
            st.write(chatbot.get_send_msg(question_2))
            st.subheader("부족한 재료")
            recipe_ingredient = pro_recipe[pro_recipe["CKG_NM"] == menu][
                "CKG_MTRL_CN"
            ].values[0]
            current_ingredient_status = check_ingredient_status(model)
            missing_ingredients = identify_necessary_ingredients(
                recipe_ingredient, current_ingredient_status
            )
            for missing_ingredient in missing_ingredients:
                st.write("- " + missing_ingredient)
        else:
            st.warning("현재 이 레시피는 준비되어 있지 않습니다.")


def main():
    option = st.sidebar.selectbox("Menu", ("홈", "회원 등록", "서비스"))
    if option == "홈":
        st.image("./streamlit_img/background.png", use_column_width=True)
        st.image("./streamlit_img/title.png", use_column_width=True)
    if option == "회원 등록":
        register_member()
    if option == "서비스":
        member = pd.read_csv("members.csv")
        st.header("메뉴 :green[추천] 서비스")
        selected_user = st.selectbox("추천 받을 사용자 선택", member["name"])
        if selected_user:
            st.subheader(f"{selected_user}님을 위한 추천 음식")
            member_index = member.index[member["name"] == selected_user].tolist()
            select_member_df = member.iloc[member_index]
            food_recommendation(select_member_df)
        st.divider()
        st.header(":green[맞춤 레시피] 제공 서비스")
        selected_users = st.multiselect("식사 인원 선택", member["name"])
        result_df = member[member["name"].isin(selected_users)]
        if selected_users:
            predict(result_df)
        else:
            st.warning("인원을 선택해주세요.")


if __name__ == "__main__":
    main()
