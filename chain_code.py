import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

DIRECOES = [(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1)]

def extrair_chain(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return None, []
    contorno = max(contornos, key=cv2.contourArea).squeeze()
    if (contorno[0]==contorno[-1]).all():
        contorno = contorno[:-1]
    codigos = []
    for i in range(len(contorno)-1):
        dx = np.sign(contorno[i+1][0] - contorno[i][0])
        dy = np.sign(contorno[i+1][1] - contorno[i][1])
        try:
            codigos.append(DIRECOES.index((dx,dy)))
        except ValueError:
            d = min(range(len(DIRECOES)), key=lambda k: (DIRECOES[k][0]-dx)**2 + (DIRECOES[k][1]-dy)**2)
            codigos.append(d)
    return tuple(contorno[0]), codigos

def run(mask_path, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    inicio, codigos = extrair_chain(mask)
    if inicio is None:
        print("Nenhum contorno encontrado.")
        return
    print("Início:", inicio, "Comprimento:", len(codigos))
    pontos = [inicio]
    for c in codigos:
        dx,dy = DIRECOES[c]
        x,y = pontos[-1]
        pontos.append((x+dx, y+dy))
    pontos = np.array(pontos)
    img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for p in pontos:
        cv2.circle(img, tuple(int(v) for v in p), 1, (0,0,255), -1)
    base = Path(mask_path).stem
    cv2.imwrite(f"{out_dir}/{base}_chain_overlay.png", img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Contorno (Chain Code)")
    plt.axis('off')
    plt.show()
    print("Primeiros 20 códigos:", codigos[:20])

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python src/chain_code.py outputs/imagem1_otsu_mask.png")
    else:
        run(sys.argv[1])
