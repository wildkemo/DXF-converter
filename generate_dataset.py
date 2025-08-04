import os
import random
import cv2
import numpy as np

# Create folders
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/labels", exist_ok=True)

def draw_random_line(img):
    pt1 = tuple(np.random.randint(0, img.shape[1], size=2))
    pt2 = tuple(np.random.randint(0, img.shape[0], size=2))
    cv2.line(img, pt1, pt2, (255, 255, 255), 2)
    return f"LINE {pt1[0]},{pt1[1]} {pt2[0]},{pt2[1]}"

def draw_random_circle(img):
    center = tuple(np.random.randint(50, 200, size=2))
    radius = np.random.randint(20, 50)
    cv2.circle(img, center, radius, (255, 255, 255), 2)
    return f"CIRCLE {center[0]},{center[1]} {radius}"

def generate_sample(idx):
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    commands = []

    for _ in range(random.randint(1, 5)):
        if random.random() > 0.5:
            commands.append(draw_random_line(img))
        else:
            commands.append(draw_random_circle(img))

    # Save image
    image_path = f"data/images/{idx:04d}.png"
    label_path = f"data/labels/{idx:04d}.txt"

    cv2.imwrite(image_path, img)

    with open(label_path, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")

for i in range(100):  # Generate 100 samples
    generate_sample(i)

print("✅ Dataset generated.")
