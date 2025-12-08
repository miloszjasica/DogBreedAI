import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from tqdm import tqdm
import time

raw_dir = "../dog_classifier/data/raw"
BREEDS = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
print(len(BREEDS), BREEDS[:10])
IMAGES_PER_BREED = 35
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

os.makedirs(raw_dir, exist_ok=True)

def fetch_image_urls(query, max_links=200):
    search_url = f"https://www.bing.com/images/search?q={quote(query)}&form=HDRSC2&first=1&tsc=ImageBasicHover"
    urls = set()
    try:
        resp = requests.get(search_url, headers=HEADERS)
        soup = BeautifulSoup(resp.text, "html.parser")
        for img_tag in soup.find_all("a", class_="iusc"):
            try:
                m = img_tag.get("m")
                if not m:
                    continue
                m = eval(m)
                u = m["murl"]
                urls.add(u)
                if len(urls) >= max_links:
                    break
            except:
                continue
    except Exception as e:
        print("Download error", e)
    return list(urls)

def download_images(urls, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, url in enumerate(tqdm(urls)):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=5)
            ext = url.split(".")[-1].split("?")[0]
            if ext.lower() not in ["jpg", "jpeg", "png"]:
                ext = "jpg"
            file_path = os.path.join(save_path, f"{i}.{ext}")
            with open(file_path, "wb") as f:
                f.write(resp.content)
            time.sleep(0.33)
        except:
            continue

if __name__ == "__main__":
    for breed in BREEDS:
        print(f"\nDownloading {IMAGES_PER_BREED} per breed: {breed}")
        query = breed.split("-", 1)[1]
        urls = fetch_image_urls(query, IMAGES_PER_BREED)
        download_images(urls, os.path.join(raw_dir, breed))
    print("\n Download completed")
