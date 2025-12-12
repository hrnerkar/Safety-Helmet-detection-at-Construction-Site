import os
import cv2

# Base paths
BASE_PATH = os.path.abspath("dataset")
OUTPUT_PATH = os.path.abspath("cnn_dataset")

# Make output folders
for split in ['train', 'valid']:
    for cls in ['helmet', 'no_helmet']:
        os.makedirs(os.path.join(OUTPUT_PATH, split, cls), exist_ok=True)

# Class map
class_map = {
    0: "helmet",
    1: "no_helmet"
}

def extract_and_save(split):
    img_dir = os.path.join(BASE_PATH, split, 'images')
    label_dir = os.path.join(BASE_PATH, split, 'labels')
    output_dir = os.path.join(OUTPUT_PATH, split)

    # ✅ Check if label directory exists
    if not os.path.exists(label_dir):
        print(f"❌ Labels folder does not exist: {label_dir}")
        return

    for filename in os.listdir(label_dir):
        label_path = os.path.join(label_dir, filename)
        img_path = os.path.join(img_dir, filename.replace('.txt', '.jpg'))

        if not os.path.exists(img_path):
            print(f"⚠️ Missing image: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Unable to load image: {img_path}")
            continue
        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            cls, x, y, box_w, box_h = map(float, line.strip().split())
            x1 = int((x - box_w / 2) * w)
            y1 = int((y - box_h / 2) * h)
            x2 = int((x + box_w / 2) * w)
            y2 = int((y + box_h / 2) * h)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            class_name = class_map[int(cls)]
            out_path = os.path.join(output_dir, class_name, f"{filename[:-4]}_{i}.jpg")
            cv2.imwrite(out_path, crop)

# Run extraction
extract_and_save("train")
extract_and_save("valid")
print("✅ Crop extraction complete.")
