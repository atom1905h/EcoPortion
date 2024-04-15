import streamlit as st
import csv


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


def main():
    option = st.sidebar.selectbox("Menu", ("회원 등록", "페이지2", "페이지3"))

    if option == "회원 등록":
        register_member()


if __name__ == "__main__":
    main()
