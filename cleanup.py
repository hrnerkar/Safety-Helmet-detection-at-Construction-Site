import os
import time

def delete_old_images(folder_path, days=30):
    now = time.time()
    cutoff = now - days * 86400  # 30 days

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            if os.path.getmtime(filepath) < cutoff:
                os.remove(filepath)
                print(f"[INFO] Deleted old file: {filename}")
