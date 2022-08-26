from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
import os


def test_check_plagiarism():
    driver = webdriver.Chrome(
        service=ChromeService(executable_path=ChromeDriverManager().install())
    )

    driver.get("http://localhost:3000")

    file_input = driver.find_element(By.ID, value="dropzone")

    examples_dir_path = os.path.abspath("./examples")
    file_input.send_keys(f"{examples_dir_path}/suspicious-document00001.txt")

    verify_button = driver.find_element(By.CLASS_NAME, "primary-button")
    verify_button.click()

    sleep(10)
    driver.quit()


test_check_plagiarism()
