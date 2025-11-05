import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from pathlib import Path

def run(image_path, K=3, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,_ = img_rgb.shape
    X = img_rgb.reshape((-1,3)).astype('float32')
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
    centers = kmeans.cluster_centers_.astype('uint8')
    labels = kmeans.labels_
    seg = centers[labels].reshape((h,w,3))
    
    base = Path(image_path).stem
    cv2.imwrite(f"{out_dir}/{base}_kmeans_K{K}.png", cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(img_rgb); plt.title('Original'); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(seg); plt.title(f'K-Means (K={K})'); plt.axis('off')
    plt.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python src/kmeans.py data/imagem1.jpg")
    else:
        run(sys.argv[1])
