import meraki
import yaml
from datetime import datetime, timedelta
import urllib.request


def save_image_from_url(url, file_path):
    try:
        with urllib.request.urlopen(url) as response:
            image_data = response.read()

        with open(file_path, "wb") as file:
            file.write(image_data)

        print("이미지를 성공적으로 저장했습니다:", file_path)

    except Exception as e:
        print("이미지를 저장하는 중에 오류가 발생했습니다:", e)


def time_slot():
    current_time = datetime.now()
    previous_time = current_time - timedelta(hours=12)
    previous_time_str = previous_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    return previous_time_str


def check_ingredient_status(model):
    with open("secret.yaml", encoding="UTF-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    API_KEY = cfg["meraki_api"]
    serial = cfg["meraki_serial"]
    dashboard = meraki.DashboardAPI(API_KEY)
    timestamp = time_slot()

    response = dashboard.camera.generateDeviceCameraSnapshot(
        serial, timestamp=timestamp, fullframe=False
    )
    url = response["url"]
    file_path = "meraki_image/image.jpg"
    save_image_from_url(url, file_path)
    result = model.predict(
        file_path, imgsz=640, conf=0.5
    )  # project='runs',name='detect
    ingredient = []

    for r in result:
        class_dic = r.names
        class_list = r.boxes.cls.tolist()
        for i in class_list:
            ingredient.append(class_dic[i])

    return ingredient


def identify_necessary_ingredients(recipe_ingredient, current_ingredient_status):
    recipe_ingredient = recipe_ingredient.split(" ")
    result = [
        value for value in recipe_ingredient if value not in current_ingredient_status
    ]

    return result
