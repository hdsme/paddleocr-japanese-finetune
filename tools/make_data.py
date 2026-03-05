import os
import pandas as pd
import cv2

csv_path = "/workspace/data/input/harika-20251201/001/OUT.csv"
base_dir = "/workspace/data/input/harika-20251201/001"

save_dir = "/workspace/paddleocr-japanese-finetune/data/image"
label_file = "/workspace/paddleocr-japanese-finetune/data/label.txt"

df = pd.read_csv(csv_path, header=None, encoding="cp932")
df.fillna("", inplace=True)
df.columns = ["filename", "kanji_text", "kana_text"]


katakana_crop = {
    "top": 0.05,
    "bottom": 0.45,
    "left": 0.23,
    "right": 1.0
}

kanji_crop = {
    "top": 0.35,
    "bottom": 0.9,
    "left": 0.2,
    "right": 0.9
}

def crop_image(img, crop_cfg):
    h, w = img.shape[:2]

    top = int(h * crop_cfg["top"])
    bottom = int(h * crop_cfg["bottom"])
    left = int(w * crop_cfg["left"])
    right = int(w * crop_cfg["right"])

    return img[top:bottom, left:right]
with open(label_file, "a", encoding="utf-8") as label_out:
    for _, row in df.iterrows():
        filename = row["filename"].replace('.tif', '.jpg')
        full_path = os.path.join(base_dir, filename)
        kanji_text = row["kanji_text"].strip().replace(" ", "").replace("　", "")
        kana_text = row["kana_text"].strip().replace(" ", "").replace("　", "")

        print(f"File: {full_path}, Kanji: {kanji_text}, Kana: {kana_text}")
        
        img = cv2.imread(full_path)

        if img is None:
            print("Cannot read:", full_path)
            continue
        
        if kana_text != "":
            katakana_img = crop_image(img, katakana_crop)
            katakana_name = f"katakana-{filename}"
            katakana_path = os.path.join(save_dir, katakana_name)
            cv2.imwrite(katakana_path, katakana_img)

            label_out.write(f"{katakana_name} {kana_text}\n")

        if kanji_text != "":
            # kanji crop
            kanji_img = crop_image(img, kanji_crop)
            kanji_name = f"kanji-{filename}"
            kanji_path = os.path.join(save_dir, kanji_name)
            cv2.imwrite(kanji_path, kanji_img)
            label_out.write(f"{kanji_name} {kanji_text}\n")

    
        print("Saved:", katakana_name, kanji_name)