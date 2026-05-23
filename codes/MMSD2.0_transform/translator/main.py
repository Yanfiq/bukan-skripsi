from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import fake_useragent

def main():
    # preparing the texts
    #read the text from the file
    with open("texts.txt", "r") as f:
        texts = f.read().splitlines()
    target_file = open("translations.txt", "a")
    failed_file = open("failed_translations.txt", "a")
    
    print(f"Total texts: {len(texts)}")
    print(f"Sample text: {texts[0]}")

    # preparing driver
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
        retry_attempts = 3
        is_success = False
        for attempt in range(retry_attempts):
            try:
                print(f"Attempt {attempt +1}")
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, source_xpath)))
                input_element = driver.find_element(By.XPATH, source_xpath)
                input_element.send_keys(text)

                # When the target has the share button, the translation is finished
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, share_target_xpath)))

                target_element = driver.find_element(By.XPATH, target_xpath)
                translation = target_element.text

                print(f"Original: {text}")
                print(f"Translated: {translation}")
                # save the translation to the file
                target_file.write(f"{translation}\n")
                
                # random delay
                time.sleep(random.uniform(3, 7))

                clear_element = driver.find_element(By.XPATH, clear_xpath)
                clear_element.click()
                is_success = True
                break
            except Exception as e:
                print(f"Error: {e}")
                driver.get(f"https://www.deepl.com/en/translator")
                continue
        if not is_success:
            print(f"Failed to translate: {text}")
            failed_file.write(f"{text}\n")
    target_file.close()
    failed_file.close()
    driver.quit()


if __name__ == "__main__":
    main()
