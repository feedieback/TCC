"""
Teste do Módulo 3 — gera imagens sintéticas de plantação e valida todos os índices.
Execute: python test_vegetation.py
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from vegetation_analysis import VegetationAnalyzer, INDICES


def create_synthetic_plantation(width=600, height=400, seed=42):
    """
    Gera imagem sintética simulando uma plantação com zonas de saúde variada:
      - Plantas saudáveis (verde vibrante)
      - Plantas estressadas (amarelo-verde)
      - Plantas doentes / solo exposto (marrom-bege)
      - Fundo de solo (marrom)
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Solo base (marrom)
    img[:] = [101, 67, 33]

    def add_plant_patch(cx, cy, radius, health):
        """Desenha mancha circular de planta com nível de saúde 0-1."""
        Y, X = np.ogrid[:height, :width]
        dist  = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask  = dist < radius
        noise = rng.integers(-15, 15, (height, width, 3))

        if health > 0.7:       # saudável — verde vibrante
            base = np.array([40, 140, 50], dtype=np.int32)
        elif health > 0.45:    # moderado — verde-amarelado
            base = np.array([80, 130, 60], dtype=np.int32)
        elif health > 0.2:     # estressado — amarelo-esverdeado
            base = np.array([130, 140, 40], dtype=np.int32)
        else:                  # crítico — marrom-esverdeado
            base = np.array([120, 90, 30], dtype=np.int32)

        for c in range(3):
            channel = img[:, :, c].astype(np.int32)
            channel[mask] = np.clip(base[c] + noise[:, :, c][mask], 0, 255)
            img[:, :, c] = channel.astype(np.uint8)

    # Zona saudável — canto superior esquerdo
    for _ in range(30):
        cx = rng.integers(20, 250)
        cy = rng.integers(20, 180)
        r  = rng.integers(15, 45)
        add_plant_patch(cx, cy, r, health=rng.uniform(0.70, 1.0))

    # Zona moderada — centro
    for _ in range(20):
        cx = rng.integers(200, 400)
        cy = rng.integers(100, 300)
        r  = rng.integers(12, 35)
        add_plant_patch(cx, cy, r, health=rng.uniform(0.40, 0.70))

    # Zona crítica — canto inferior direito
    for _ in range(18):
        cx = rng.integers(380, 580)
        cy = rng.integers(250, 390)
        r  = rng.integers(10, 30)
        add_plant_patch(cx, cy, r, health=rng.uniform(0.0, 0.35))

    return img


def test_all_indices():
    print("\n" + "=" * 56)
    print("  TESTE DO MÓDULO 3 — TODOS OS ÍNDICES")
    print("=" * 56)

    # Gera e salva imagem sintética
    os.makedirs("test_output", exist_ok=True)
    img_bgr = create_synthetic_plantation()
    img_path = "test_output/plantacao_sintetica.jpg"
    cv2.imwrite(img_path, img_bgr)
    print(f"\n[✓] Imagem sintética gerada: {img_path}")

    results = {}
    for idx_name in INDICES:
        print(f"\n  Testando índice: {idx_name}...")
        try:
            analyzer = VegetationAnalyzer(index_name=idx_name)
            res = analyzer.analyze_image(img_path)
            m   = res["metrics"]
            results[idx_name] = m

            analyzer.print_summary()
            analyzer.save_report(output_dir="test_output", prefix="teste")
            print(f"  [✓] {idx_name}: saúde={m['health_pct']:.1f}%  cobertura={m['green_coverage_pct']:.1f}%")
        except Exception as e:
            print(f"  [✗] {idx_name}: ERRO — {e}")

    print("\n" + "=" * 56)
    print("  RESUMO COMPARATIVO")
    print("=" * 56)
    print(f"  {'Índice':<8} {'Saúde%':>8} {'Cobertura%':>12} {'Média':>10} {'σ':>8}")
    print("  " + "-" * 50)
    for idx_name, m in results.items():
        print(
            f"  {idx_name:<8} {m['health_pct']:>8.1f} "
            f"{m['green_coverage_pct']:>12.1f} "
            f"{m['mean_index']:>+10.4f} "
            f"{m['std_index']:>8.4f}"
        )
    print("=" * 56)
    print("\n[✓] Todos os arquivos de saída salvos em: test_output/\n")


if __name__ == "__main__":
    test_all_indices()  