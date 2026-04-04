from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import fake_useragent

texts = [
    "Hello, what is your name?",
    "My name is Skripsi, I like to make students suffer",
    "Why is that?",
    "Because suffering is the key to success",
    "I see, but i can make you suffer more",
    "Wow, what is your name?",
    "My name is ligma",
    "What is ligma?",
    "Ligmaballz"
]

def main():
    options = Options()
    options.binary_location = "/usr/bin/google-chrome-stable" 
    options.add_experimental_option("debuggerAddress", "localhost:9222")
    options.add_argument(f"user-agent={fake_useragent.UserAgent().random}")

    driver = webdriver.Chrome(options=options)

    driver.get(f"https://www.deepl.com/en/translator")

    source_xpath = "//d-textarea[@name='source']"
    target_xpath = """//*[@id="headlessui-tabs-panel-:Rqn8psqkukmmfnmlajsq:"]/div/div/div/div/div/div[1]/div[1]/div/div[2]/div[2]/div[1]/div/div[1]/d-textarea/div/p"""
    share_target_xpath = """/html/body/div[2]/div[2]/div/div[2]/div[1]/div[2]/main/div[2]/nav/div/div[2]/div/div/div/div/div/div/div[1]/div[1]/div/div[2]/div[2]/div[2]/div/div[6]/div[1]/span/span/span/span/span/span/button"""
    clear_xpath = """/html/body/div[2]/div[2]/div/div[2]/div[1]/div[2]/main/div[2]/nav/div/div[2]/div/div/div/div/div/div/div[1]/div[1]/div/div[2]/div[1]/div[1]/div/div/div[1]/div/span/span/span/button"""

    for text in texts:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, source_xpath)))
        input_element = driver.find_element(By.XPATH, source_xpath)
        input_element.send_keys(text)

        # When the target has the share button, the translation is finished
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, share_target_xpath)))

        target_element = driver.find_element(By.XPATH, target_xpath)
        translation = target_element.text

        print(f"Original: {text}")
        print(f"Translated: {translation}")

        # random delay
        time.sleep(random.uniform(3, 10))

        clear_element = driver.find_element(By.XPATH, clear_xpath)
        clear_element.click()
    driver.quit()


if __name__ == "__main__":
    main()
