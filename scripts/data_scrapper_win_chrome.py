import os
import time
import requests
from tqdm import tqdm
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

RAW_DIR = "../dog_classifier/data/raw"
BREEDS = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
IMAGES_PER_BREED = 200

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver

def scroll_page(driver, num_scrolls=15):
    for _ in range(num_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

def fetch_image_urls(query, max_links):
    search_url = f"https://www.bing.com/images/search?q={quote(query)}"
    driver = init_driver()
    driver.get(search_url)
    time.sleep(2)

    scroll_page(driver, num_scrolls=15)

    image_elements = driver.find_elements(By.CSS_SELECTOR, "img.mimg")
    urls = set()

    for img in image_elements:
        src = img.get_attribute("src")
        if src and "http" in src:
            urls.add(src)
        if len(urls) >= max_links:
            break

    driver.quit()
    return list(urls)

def download_images(urls, save_path):
    os.makedirs(save_path, exist_ok=True)

    for i, url in enumerate(tqdm(urls)):
        try:
            r = requests.get(url, headers=HEADERS, timeout=4)
            ext = "jpg"
            file_path = os.path.join(save_path, f"{i}.{ext}")

            with open(file_path, "wb") as f:
                f.write(r.content)

        except Exception:
            continue

if __name__ == "__main__":
    for breed in BREEDS:
        breed_name = breed.split("-", 1)[1]
        print(f"\nFetching for: {breed_name}")

        urls = fetch_image_urls(breed_name, IMAGES_PER_BREED)
        download_images(urls, os.path.join(RAW_DIR, breed))

    print("\nDONE!")
