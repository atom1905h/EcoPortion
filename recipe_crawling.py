from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.keys import Keys


def recipe_crawling(recipe, menu):
    browser = webdriver.Chrome()
    url = "https://www.10000recipe.com/index.html"
    browser.get(url)
    time.sleep(0.25)
    menu_data = recipe[recipe["CKG_NM"] == menu]
    menu_id = str(menu_data["RCP_SNO"].values[0])
    input_text = browser.find_element("css selector", "input#srhRecipeText")
    input_text.send_keys(menu_id)
    input_text.send_keys(Keys.RETURN)
    time.sleep(0.25)
    menu_link = browser.find_elements("css selector", "a.common_sp_link")[0]
    menu_link.click()
    time.sleep(0.25)
    html = browser.page_source
    soup = BeautifulSoup(html, "html.parser")
    step_list = soup.find_all(
        class_=lambda x: x and x.startswith("view_step_cont media step")
    )
    step = []
    for i in range(len(step_list)):
        step_recipe = step_list[i].select("div.media-body")[0].text
        step.append(step_recipe)
    formatted_sentences = "\n".join([f"{i+1}. {ste}" for i, ste in enumerate(step)])

    return formatted_sentences
