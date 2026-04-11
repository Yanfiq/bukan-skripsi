from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import fake_useragent
import urllib.request

def main():
    # preparing the texts
    #read the text from the file
    with open("links.txt", "r") as f:
        links = f.read().splitlines()
    
    print(f"Total links: {len(links)}")
    print(f"Sample link: {links[0]}")

    # preparing driver
    options = Options()
    options.binary_location = "/usr/bin/google-chrome-stable" 
    # options.add_experimental_option("debuggerAddress", "localhost:9222")
    options.add_argument(f"user-agent={fake_useragent.UserAgent().random}")

    driver = webdriver.Chrome(options=options)

    for link in links:
        try:
            driver.get(link)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "img")))
            images = driver.find_elements(By.TAG_NAME, "img")
            for image in images:
                src = image.get_attribute("src")
                if src and src.startswith("http"):
                    # get the filename
                    filename = src.split("/")[-1]
                    print(src, filename)
                    # urllib.request.urlretrieve(src, filename)
        except Exception as ex:
            print(ex)

    driver.quit()


if __name__ == "__main__":
    main()
