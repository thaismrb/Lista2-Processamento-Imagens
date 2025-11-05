import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

def run(image_path, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    limiar, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Limiar de Otsu encontrado: {limiar}")
    
    base = Path(image_path).stem
    cv2.imwrite(f"{out_dir}/{base}_otsu_mask.png", mask)
    segmented = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(f"{out_dir}/{base}_otsu_segmented.png", segmented)
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Original'); plt.axis('off')
    plt.subplot(1,3,2); plt.hist(gray.ravel(),256,[0,256]); plt.axvline(limiar,color='r'); plt.title('Histograma')
    plt.subplot(1,3,3); plt.imshow(mask, cmap='gray'); plt.title('Otsu - Binária'); plt.axis('off')
    plt.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python src/otsu.py data/imagem1.jpg")
    else:
        run(sys.argv[1])
