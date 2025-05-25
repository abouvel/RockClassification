from icrawler.builtin import BingImageCrawler
import os
import time
import hashlib
import subprocess

# === CONFIG ===
shape_queries = {
    "cube": [
        "cube shaped rock", "blocky rock", "angular stone",
        "jointed granite block", "cuboid sandstone"
    ],
    "cylinder": [
        "rock core drill sample", "cylindrical basalt column",
        "drilled stone cylinder", "stone column geology",
        "cylindrical sedimentary rock"
    ],
    "ellipsoid": [
        "oval river rock", "egg shaped stone", "ellipsoid pebble",
        "smooth elliptical rock", "elongated stone pebble"
    ],
    "sphere": [
        "round stone on beach", "spherical river rock",
        "natural ball shaped rock", "smooth round pebble",
        "glacially rounded boulder"
    ]
}

output_root = "/home/abouvel/claws/AI-NLP-24-25/IMGDetector/shapes_dataset/realShapes"
target_per_class = 2500
images_per_query = 500
max_retries_per_query = 3

# === UTILS ===
def valid_image_files(folder):
    return [f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.getsize(os.path.join(folder, f)) > 10 * 1024]

def deduplicate_images(folder):
    hashes = {}
    duplicates = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
            if img_hash in hashes:
                duplicates.append(fpath)
            else:
                hashes[img_hash] = fpath
        except Exception:
            duplicates.append(fpath)
    for dup in duplicates:
        os.remove(dup)
    return len(duplicates)

def strip_metadata(folder):
    print("🧼 Stripping metadata...")
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        subprocess.run(["mogrify", "-strip", os.path.join(folder, ext)], shell=False)

# === DOWNLOAD LOOP ===
for shape, queries in shape_queries.items():
    shape_dir = os.path.join(output_root, shape)
    os.makedirs(shape_dir, exist_ok=True)

    print(f"\n📦 Target: {target_per_class} images for '{shape}'")
    total_downloaded = len(valid_image_files(shape_dir))
    query_index = 0

    while total_downloaded < target_per_class and query_index < len(queries):
        query = queries[query_index]
        remaining = target_per_class - total_downloaded
        num_to_fetch = min(images_per_query, remaining)

        for attempt in range(max_retries_per_query):
            print(f"🔍 Query {query_index + 1} (attempt {attempt+1}): '{query}' for {num_to_fetch} images")
            crawler = BingImageCrawler(storage={"root_dir": shape_dir})
            try:
                crawler.crawl(keyword=query, max_num=num_to_fetch)
            except Exception as e:
                print(f"⚠️ Error during crawl: {e}")
                break

            time.sleep(1)
            strip_metadata(shape_dir)
            removed = deduplicate_images(shape_dir)
            total_downloaded = len(valid_image_files(shape_dir))
            print(f"✅ Total downloaded so far: {total_downloaded} (Removed {removed} duplicates)")

            if total_downloaded >= target_per_class:
                break

        query_index += 1

    print(f"✅ Final total for '{shape}': {total_downloaded}")
    if total_downloaded < target_per_class:
        print(f"⚠️ Still under target. Add more queries for '{shape}' if needed.")
