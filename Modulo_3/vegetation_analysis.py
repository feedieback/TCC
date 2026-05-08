"""
Módulo 3 — Sistema Inteligente de Monitoramento Agrícola
Análise de Saúde da Vegetação via Índices RGB

Autor: Victor Augusto de Oliveira
Orientador: Prof. Dr Maurício Acconcia Dias
FHO — Engenharia da Computação, 2025

Descrição:
    Processa imagens RGB de plantações e calcula índices de vegetação
    (VARI, ExG, GLI, NGRDI, VEG, ExGR, TomatoVI) para gerar mapas de saúde,
    métricas quantitativas e diagnósticos automáticos.

    Modo TOMATE (--crop tomato):
        Segmenta a imagem em HSV antes do cálculo, separando folhagem, frutos
        e solo. Avalia a saúde apenas sobre a máscara de folhagem verde e usa
        índices adaptados a culturas com frutos vermelhos (ExGR, TomatoVI).

    Classificação automática por saúde:
        saudavel  → health_pct >= 60
        alerta    → 35 <= health_pct < 60
        critico   → health_pct < 35

    Todas as análises são registradas em log.txt no diretório de saída.

Uso:
    python vegetation_analysis.py --image imagem.jpg
    python vegetation_analysis.py --image imagem.jpg --index ExGR --crop tomato
    python vegetation_analysis.py --folder pasta/imagens/ --crop tomato
    python vegetation_analysis.py --camera 0 --crop tomato
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from matplotlib.gridspec import GridSpec
import argparse
import os
import json
import datetime
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Classificação de saúde e logger
# ---------------------------------------------------------------------------

# Limites de classificação (ajustáveis)
CLASSIFY_SAUDAVEL = 60   # health_pct >= 60  → saudavel
CLASSIFY_ALERTA   = 35   # 35 <= pct < 60    → alerta
#                          pct < 35           → critico


def classify_health(health_pct: float) -> str:
    """
    Converte o percentual de saúde (0-100) em uma classificação textual.

    Retorna
    -------
    'saudavel' | 'alerta' | 'critico'
    """
    if health_pct >= CLASSIFY_SAUDAVEL:
        return "saudavel"
    elif health_pct >= CLASSIFY_ALERTA:
        return "alerta"
    else:
        return "critico"


class HealthLogger:
    """
    Registra os resultados de cada análise em um arquivo log.txt.

    Formato de cada linha:
        nome_arquivo.png - classificacao  [saude: XX.X%]  [indice]  [timestamp]

    Parâmetros
    ----------
    log_path : str
        Caminho completo para o arquivo log.txt.
    index_name : str
        Nome do índice usado nas análises (gravado no cabeçalho da sessão).
    """

    def __init__(self, log_path: str, index_name: str = "VARI"):
        self.log_path   = log_path
        self.index_name = index_name
        self._session_started = False

    def _start_session(self):
        """Escreve o cabeçalho da sessão no log na primeira entrada."""
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"  Sessão iniciada em: {ts}\n")
            f.write(f"  Índice utilizado  : {self.index_name}\n")
            f.write(f"{'='*60}\n")
        self._session_started = True

    def record(self, filename: str, health_pct: float, classification: str):
        """
        Grava uma linha de resultado no log.

        Parâmetros
        ----------
        filename       : nome do arquivo ou identificador do frame
        health_pct     : percentual de saúde calculado (0-100)
        classification : 'saudavel' | 'alerta' | 'critico'
        """
        if not self._session_started:
            self._start_session()

        ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"{filename} - {classification}"
            f"  [saude: {health_pct:.1f}%]"
            f"  [{self.index_name}]"
            f"  [{ts}]\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def summary(self, counts: dict):
        """Grava um resumo da sessão (total por classificação)."""
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- Resumo da sessão ({ts}) ---\n")
            for k, v in counts.items():
                f.write(f"  {k:<10}: {v} imagem(ns)\n")
            f.write("\n")

def compute_vari(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    VARI — Vegetation Atmospherically Resistant Index
    Robusto a variações de iluminação; ideal para câmeras RGB convencionais.
    Fórmula: (G - R) / (G + R - B)
    Intervalo típico: [-1, 1]
    """
    denom = g + r - b
    with np.errstate(invalid="ignore", divide="ignore"):
        index = np.where(denom != 0, (g - r) / denom, 0.0)
    return np.clip(index, -1.0, 1.0)


def compute_exg(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    ExG — Excess Green Index
    Realça pixels com forte componente verde; eficaz para segmentar vegetação.
    Fórmula: 2G_n - R_n - B_n  (canais normalizados pela soma)
    Intervalo típico: [-1, 1]
    """
    total = r + g + b
    with np.errstate(invalid="ignore", divide="ignore"):
        rn = np.where(total > 0, r / total, 0.0)
        gn = np.where(total > 0, g / total, 0.0)
        bn = np.where(total > 0, b / total, 0.0)
    return np.clip(2 * gn - rn - bn, -1.0, 1.0)


def compute_gli(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    GLI — Green Leaf Index
    Combina verde e vermelho para distinguir folhas verdes de solo e fundo.
    Fórmula: (2G - R - B) / (2G + R + B)
    Intervalo típico: [-1, 1]
    """
    num = 2.0 * g - r - b
    denom = 2.0 * g + r + b
    with np.errstate(invalid="ignore", divide="ignore"):
        index = np.where(denom > 0, num / denom, 0.0)
    return np.clip(index, -1.0, 1.0)


def compute_ngrdi(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    NGRDI — Normalized Green-Red Difference Index
    Sensível ao estresse hídrico e ao teor de clorofila.
    Fórmula: (G - R) / (G + R)
    Intervalo típico: [-1, 1]
    """
    denom = g + r
    with np.errstate(invalid="ignore", divide="ignore"):
        index = np.where(denom > 0, (g - r) / denom, 0.0)
    return np.clip(index, -1.0, 1.0)


def compute_veg(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    VEG — Vegetative Index (Hague et al., 2006)
    Minimiza o efeito do solo; realça diferenças sutis de clorofila.
    Fórmula: G / (R^0.667 × B^0.333)
    Normalizado para [0, 1] dividindo por 4.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.power(np.maximum(r, 1e-6), 0.667) * \
                np.power(np.maximum(b, 1e-6), 0.333)
        index = np.where(denom > 0, g / denom, 0.0)
    return np.clip(index / 4.0, 0.0, 1.0)


def compute_exgr(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    ExGR — Excess Green minus Excess Red
    Cancela o sinal de frutos e solo avermelhado, realçando apenas folhagem.
    Ideal para culturas com frutos vermelhos como tomate.
    Fórmula: (2G - R - B) - (1.4R - G)  → simplificado: 3G - 2.4R - B
    Normalizado para [-1, 1].
    Referência: Meyer & Neto (2008)
    """
    total = r + g + b
    with np.errstate(invalid="ignore", divide="ignore"):
        rn = np.where(total > 0, r / total, 0.0)
        gn = np.where(total > 0, g / total, 0.0)
        bn = np.where(total > 0, b / total, 0.0)
    exg = 2 * gn - rn - bn
    exr = 1.4 * rn - gn
    return np.clip(exg - exr, -1.0, 1.0)


def compute_tomato_vi(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    TomatoVI — Vegetative Index adaptado para plantações de tomate.
    Maximiza resposta de folhas verdes e suprime frutos vermelhos e solo.
    Fórmula: (G - 0.9*R - 0.1*B) / (G + 0.9*R + 0.1*B)
    Intervalo: [-1, 1]  — valores positivos = folhagem saudável.
    """
    num   = g - 0.9 * r - 0.1 * b
    denom = g + 0.9 * r + 0.1 * b
    with np.errstate(invalid="ignore", divide="ignore"):
        index = np.where(denom > 0, num / denom, 0.0)
    return np.clip(index, -1.0, 1.0)


# Registro central dos índices disponíveis
INDICES = {
    "VARI": {
        "func": compute_vari,
        "range": (-1.0, 1.0),
        "description": "Vegetation Atmospherically Resistant Index",
        "reference": "Gitelson et al. (2002)",
        "green_thresh": 0.10,
        "crop_note": "Genérico — distorcido por frutos vermelhos",
    },
    "ExG": {
        "func": compute_exg,
        "range": (-1.0, 1.0),
        "description": "Excess Green Index",
        "reference": "Woebbecke et al. (1995)",
        "green_thresh": 0.10,
        "crop_note": "Genérico — distorcido por frutos vermelhos",
    },
    "GLI": {
        "func": compute_gli,
        "range": (-1.0, 1.0),
        "description": "Green Leaf Index",
        "reference": "Louhaichi et al. (2001)",
        "green_thresh": 0.05,
        "crop_note": "Genérico",
    },
    "NGRDI": {
        "func": compute_ngrdi,
        "range": (-1.0, 1.0),
        "description": "Normalized Green-Red Difference Index",
        "reference": "Tucker (1979)",
        "green_thresh": 0.05,
        "crop_note": "Genérico — distorcido por frutos vermelhos",
    },
    "VEG": {
        "func": compute_veg,
        "range": (0.0, 1.0),
        "description": "Vegetative Index (Hague)",
        "reference": "Hague et al. (2006)",
        "green_thresh": 0.30,
        "crop_note": "Genérico",
    },
    "ExGR": {
        "func": compute_exgr,
        "range": (-1.0, 1.0),
        "description": "Excess Green minus Excess Red",
        "reference": "Meyer & Neto (2008)",
        "green_thresh": 0.05,
        "crop_note": "Recomendado para tomate — cancela sinal de frutos vermelhos",
    },
    "TomatoVI": {
        "func": compute_tomato_vi,
        "range": (-1.0, 1.0),
        "description": "Vegetative Index adaptado para Tomate",
        "reference": "Oliveira (2025)",
        "green_thresh": 0.02,
        "crop_note": "Otimizado para tomate — suprime frutos e solo avermelhado",
    },
}


# ---------------------------------------------------------------------------
# Colormap personalizado: vermelho → amarelo → verde
# ---------------------------------------------------------------------------

def build_health_colormap() -> mcolors.LinearSegmentedColormap:
    """Cria o colormap de saúde: crítico (vermelho) → ótimo (verde)."""
    colors_list = [
        (0.00, "#C0392B"),   # crítico
        (0.20, "#E67E22"),   # ruim
        (0.50, "#F1C40F"),   # moderado
        (0.75, "#8ECF3B"),   # bom
        (1.00, "#27AE60"),   # ótimo
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "health", [(pos, color) for pos, color in colors_list]
    )
    return cmap


HEALTH_CMAP = build_health_colormap()


# ---------------------------------------------------------------------------
# Segmentação HSV — modo culturas com frutos coloridos (tomate, pimentão…)
# ---------------------------------------------------------------------------

class CropSegmenter:
    """
    Segmenta uma imagem BGR em regiões semanticamente distintas via HSV,
    isolando a folhagem verde de frutos vermelhos/laranjas e solo.

    Limites HSV do OpenCV: H∈[0,179], S∈[0,255], V∈[0,255]
    """

    # ── Folhagem verde (inclui folhas jovens e maduras) ───────────────────────
    FOLIAGE_HSV = [
        {"lo": ( 25,  35,  35), "hi": ( 85, 255, 255)},   # verde-amarelado → verde escuro
    ]

    # ── Frutos vermelhos e laranjas (tomate maduro e semi-maduro) ─────────────
    FRUIT_HSV = [
        {"lo": (  0,  80,  60), "hi": ( 10, 255, 255)},   # vermelho baixo
        {"lo": (165,  80,  60), "hi": (179, 255, 255)},   # vermelho alto (wrap-around)
        {"lo": ( 10,  80,  60), "hi": ( 25, 255, 255)},   # laranja-avermelhado
    ]

    # ── Solo / substrato (marrom, bege, cinza) ────────────────────────────────
    SOIL_HSV = [
        {"lo": ( 10,  15,  30), "hi": ( 30, 160, 180)},   # marrom-bege
        {"lo": (  0,   0,  30), "hi": (179,  40, 140)},   # cinza escuro / sombra
    ]

    @classmethod
    def _build_mask(cls, hsv: np.ndarray, ranges: list) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rng in ranges:
            lo  = np.array(rng["lo"], dtype=np.uint8)
            hi  = np.array(rng["hi"], dtype=np.uint8)
            mask |= cv2.inRange(hsv, lo, hi)
        # Limpeza morfológica: remove ruído e fecha buracos pequenos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    @classmethod
    def segment(cls, img_bgr: np.ndarray) -> dict:
        """
        Retorna dicionário com máscaras booleanas para cada região.

        Chaves: 'foliage', 'fruit', 'soil', 'other'
        Cada valor é um np.ndarray bool de mesma forma que a imagem.
        """
        hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        foliage  = cls._build_mask(hsv, cls.FOLIAGE_HSV).astype(bool)
        fruit    = cls._build_mask(hsv, cls.FRUIT_HSV).astype(bool)
        soil     = cls._build_mask(hsv, cls.SOIL_HSV).astype(bool)

        # "other" = tudo que não se encaixa nas categorias acima (céu, estufas…)
        classified = foliage | fruit | soil
        other      = ~classified

        total = img_bgr.shape[0] * img_bgr.shape[1]
        return {
            "foliage":       foliage,
            "fruit":         fruit,
            "soil":          soil,
            "other":         other,
            "foliage_pct":   float(np.sum(foliage)) / total * 100,
            "fruit_pct":     float(np.sum(fruit))   / total * 100,
            "soil_pct":      float(np.sum(soil))    / total * 100,
        }

    @classmethod
    def build_segment_overlay(cls, img_bgr: np.ndarray, masks: dict) -> np.ndarray:
        """
        Constrói imagem colorida mostrando cada região segmentada.
            Verde escuro → folhagem
            Laranja      → frutos
            Marrom       → solo
            Cinza        → outros
        """
        overlay = np.full_like(img_bgr, 50)   # fundo cinza escuro
        overlay[masks["foliage"]] = (34,  139,  34)   # verde floresta (BGR)
        overlay[masks["fruit"]]   = (0,   100, 230)   # laranja (BGR)
        overlay[masks["soil"]]    = (45,   82, 101)   # marrom (BGR)
        overlay[masks["other"]]   = (100, 100, 100)   # cinza médio
        # Blend leve com imagem original para manter legibilidade
        blended = cv2.addWeighted(img_bgr, 0.30, overlay, 0.70, 0)
        return blended

class VegetationAnalyzer:
    """
    Analisa a saúde da vegetação em imagens RGB usando índices espectrais.

    Parâmetros
    ----------
    index_name : str
        Índice a utilizar (VARI, ExG, GLI, NGRDI, VEG, ExGR, TomatoVI).
    max_dim : int
        Dimensão máxima para redimensionamento interno (preserva proporção).
    crop : str ou None
        Tipo de cultura para ativar segmentação HSV específica.
        'tomato' → avalia saúde apenas sobre folhagem, ignorando frutos e solo.
        None     → comportamento genérico (sem segmentação).
    """

    ZONE_THRESHOLDS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
    ZONE_LABELS     = ["Crítico", "Ruim", "Moderado", "Bom", "Ótimo"]
    ZONE_COLORS     = ["#C0392B", "#E67E22", "#F1C40F", "#8ECF3B", "#27AE60"]

    CROP_RECOMMENDED = {"tomato": "ExGR"}

    def __init__(self, index_name: str = "VARI", max_dim: int = 800,
                 crop: str = None):
        if index_name not in INDICES:
            raise ValueError(
                f"Índice '{index_name}' inválido. "
                f"Escolha entre: {list(INDICES.keys())}"
            )
        self.index_name = index_name
        self.max_dim    = max_dim
        self.crop       = crop.lower().strip() if crop else None
        self._meta      = INDICES[index_name]
        self._results   = None

        if self.crop == "tomato" and index_name not in ("ExGR", "TomatoVI", "GLI"):
            print(f"  [AVISO] Para tomate, recomenda-se --index ExGR ou TomatoVI.")
            print(f"          {index_name} pode ser distorcido pelos frutos vermelhos.")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def analyze_image(self, image_path: str) -> dict:
        """
        Processa uma imagem de arquivo e retorna o dicionário de resultados.

        Parâmetros
        ----------
        image_path : str
            Caminho para a imagem JPG/PNG/WEBP.

        Retorna
        -------
        dict com chaves: index_map, heatmap_bgr, metrics, diagnosis, metadata.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        return self._run(img_bgr, source=image_path)

    def analyze_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Processa um frame BGR (numpy array) diretamente — útil para câmera ao vivo.
        """
        return self._run(frame_bgr, source="frame")

    def save_report(self, output_dir: str = ".", prefix: str = "relatorio"):
        """
        Salva heatmap, mapa de zonas, gráfico de distribuição e JSON de métricas.
        """
        if self._results is None:
            raise RuntimeError("Execute analyze_image() ou analyze_frame() primeiro.")

        os.makedirs(output_dir, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(output_dir, f"{prefix}_{self.index_name}_{ts}")

        # Heatmap colorido
        cv2.imwrite(base + "_heatmap.png", self._results["heatmap_bgr"])

        # Mapa de zonas (segmentado em 5 faixas)
        cv2.imwrite(base + "_zonas.png", self._results["zones_bgr"])

        # Figura completa com matplotlib
        fig = self._build_figure()
        fig.savefig(base + "_relatorio.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # JSON de métricas
        report = {
            "metadata": self._results["metadata"],
            "metrics":  self._results["metrics"],
            "diagnosis": self._results["diagnosis"],
        }
        with open(base + "_metricas.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n[Módulo 3] Relatório salvo em: {output_dir}")
        print(f"  Heatmap : {base}_heatmap.png")
        print(f"  Zonas   : {base}_zonas.png")
        print(f"  Figura  : {base}_relatorio.png")
        print(f"  JSON    : {base}_metricas.json")

        return base

    def show(self):
        """Exibe a figura de análise em janela interativa."""
        if self._results is None:
            raise RuntimeError("Execute analyze_image() ou analyze_frame() primeiro.")
        fig = self._build_figure()
        plt.show()
        plt.close(fig)

    def print_summary(self):
        """Imprime resumo textual no terminal."""
        if self._results is None:
            raise RuntimeError("Execute analyze_image() ou analyze_frame() primeiro.")
        m    = self._results["metrics"]
        d    = self._results["diagnosis"]
        meta = self._results["metadata"]

        print("\n" + "=" * 56)
        print("  MÓDULO 3 — ANÁLISE DE SAÚDE DA VEGETAÇÃO")
        print("=" * 56)
        print(f"  Índice        : {self.index_name} — {self._meta['description']}")
        print(f"  Cultura       : {meta['crop'].upper()}")
        print(f"  Fonte         : {meta['source']}")
        print(f"  Resolução     : {meta['width']} × {meta['height']} px")
        print(f"  Processado em : {meta['processed_at']}")
        print("-" * 56)
        print(f"  Índice médio  : {m['mean_index']:+.4f}  (sobre folhagem)")
        print(f"  Saúde geral   : {m['health_pct']:.1f}%  →  {m['health_label']}")
        print(f"  Classificação : {m['classification'].upper()}")
        print(f"  Cobertura veg.: {m['green_coverage_pct']:.1f}%  (Bom + Ótimo)")
        print(f"  Desvio padrão : {m['std_index']:.4f}  ({m['uniformity_label']})")

        if "seg_foliage_pct" in m:
            print("-" * 56)
            print("  Segmentação HSV:")
            print(f"    Folhagem  : {m['seg_foliage_pct']:.1f}%")
            print(f"    Frutos    : {m['seg_fruit_pct']:.1f}%")
            print(f"    Solo      : {m['seg_soil_pct']:.1f}%")

        print("-" * 56)
        print("  Distribuição por zona (folhagem):")
        for label, pct in zip(self.ZONE_LABELS, m["zone_pct"]):
            bar = "█" * int(pct / 2)
            print(f"    {label:<10} {pct:5.1f}%  {bar}")
        print("-" * 56)
        print("  Diagnóstico automático:")
        for item in d:
            print(f"    [{item['level'].upper()}] {item['message']}")
        print("=" * 56 + "\n")

    # ------------------------------------------------------------------
    # Lógica interna
    # ------------------------------------------------------------------

    def _run(self, img_bgr: np.ndarray, source: str) -> dict:
        img_bgr = self._resize(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Normaliza canais para [0, 1]
        r = img_rgb[:, :, 0].astype(np.float32) / 255.0
        g = img_rgb[:, :, 1].astype(np.float32) / 255.0
        b = img_rgb[:, :, 2].astype(np.float32) / 255.0

        # Calcula índice sobre a imagem completa
        index_map = self._meta["func"](r, g, b)

        # ── Segmentação HSV (modo cultura específica) ──────────────────────────
        seg_masks   = None
        segment_bgr = None

        if self.crop in ("tomato",):
            seg_masks   = CropSegmenter.segment(img_bgr)
            segment_bgr = CropSegmenter.build_segment_overlay(img_bgr, seg_masks)

            # Máscara de folhagem: usada para calcular métricas de saúde
            foliage_mask = seg_masks["foliage"]

            # Se a folhagem for muito pequena, cai no modo genérico com aviso
            if seg_masks["foliage_pct"] < 2.0:
                print("  [AVISO] Folhagem detectada < 2% da imagem. "
                      "Usando imagem completa para análise.")
                foliage_mask = np.ones(img_bgr.shape[:2], dtype=bool)
        else:
            foliage_mask = np.ones(img_bgr.shape[:2], dtype=bool)

        # Índice normalizado para colorização (imagem completa → heatmap bonito)
        lo, hi     = self._meta["range"]
        index_norm = np.clip((index_map - lo) / (hi - lo), 0.0, 1.0)

        # Heatmap: pixels de folhagem em cor de saúde; resto em cinza escuro
        heatmap_rgb = np.full((*img_bgr.shape[:2], 3), 40, dtype=np.uint8)
        health_colors = (HEALTH_CMAP(index_norm)[:, :, :3] * 255).astype(np.uint8)
        if self.crop in ("tomato",) and seg_masks is not None:
            # Folhagem → cor de saúde; fruto → vermelho translúcido; solo → cinza
            heatmap_rgb[foliage_mask]         = health_colors[foliage_mask]
            heatmap_rgb[seg_masks["fruit"]]   = [200, 60, 40]    # laranja escuro (RGB)
            heatmap_rgb[seg_masks["soil"]]    = [100, 80, 60]    # marrom escuro
            heatmap_rgb[seg_masks["other"]]   = [70,  70, 70]    # cinza
        else:
            heatmap_rgb = health_colors

        heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

        # Mapa de zonas (apenas folhagem colorida)
        zones_bgr = self._build_zone_map(index_norm, foliage_mask,
                                         seg_masks if seg_masks else {})

        # Métricas calculadas sobre a máscara de folhagem
        metrics   = self._compute_metrics(index_map, index_norm,
                                          foliage_mask, seg_masks)
        diagnosis = self._diagnose(metrics)

        self._results = {
            "original_bgr": img_bgr,
            "index_map":    index_map,
            "index_norm":   index_norm,
            "heatmap_bgr":  heatmap_bgr,
            "zones_bgr":    zones_bgr,
            "segment_bgr":  segment_bgr,   # None se crop=None
            "seg_masks":    seg_masks,
            "metrics":      metrics,
            "diagnosis":    diagnosis,
            "metadata": {
                "source":       source,
                "index":        self.index_name,
                "crop":         self.crop or "generico",
                "width":        img_bgr.shape[1],
                "height":       img_bgr.shape[0],
                "processed_at": datetime.datetime.now().isoformat(),
            },
        }
        return self._results

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= self.max_dim:
            return img
        scale = self.max_dim / max(h, w)
        return cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    def _build_zone_map(self, index_norm: np.ndarray,
                        foliage_mask: np.ndarray = None,
                        seg_masks: dict = None) -> np.ndarray:
        """Cria mapa BGR com 5 cores sólidas por zona de saúde.
        Pixels fora da máscara de folhagem recebem cor contextual."""
        h, w       = index_norm.shape
        zone_rgb   = np.full((h, w, 3), 50, dtype=np.uint8)   # fundo escuro
        thresholds = self.ZONE_THRESHOLDS

        # Máscara padrão: imagem inteira
        if foliage_mask is None:
            foliage_mask = np.ones((h, w), dtype=bool)

        for i, color_hex in enumerate(self.ZONE_COLORS):
            zone_mask = (index_norm >= thresholds[i]) & \
                        (index_norm <  thresholds[i + 1]) & foliage_mask
            rgb = tuple(int(color_hex.lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
            zone_rgb[zone_mask] = rgb

        # Última zona (≥ 0.80) dentro da folhagem
        zone_rgb[(index_norm >= thresholds[-2]) & foliage_mask] = tuple(
            int(self.ZONE_COLORS[-1].lstrip("#")[j:j+2], 16) for j in (0, 2, 4)
        )

        # Cores contextuais para regiões não-folhagem
        if seg_masks:
            zone_rgb[seg_masks.get("fruit", np.zeros((h,w), bool))] = (200,  60,  40)
            zone_rgb[seg_masks.get("soil",  np.zeros((h,w), bool))] = (101,  82,  45)
            zone_rgb[seg_masks.get("other", np.zeros((h,w), bool))] = ( 90,  90,  90)

        return cv2.cvtColor(zone_rgb, cv2.COLOR_RGB2BGR)

    def _compute_metrics(
        self,
        index_map:    np.ndarray,
        index_norm:   np.ndarray,
        foliage_mask: np.ndarray = None,
        seg_masks:    dict       = None,
    ) -> dict:
        # Se há máscara de folhagem, calcula métricas apenas sobre ela
        if foliage_mask is not None and np.any(foliage_mask):
            idx_foliage  = index_map[foliage_mask]
            norm_foliage = index_norm[foliage_mask]
        else:
            idx_foliage  = index_map.flatten()
            norm_foliage = index_norm.flatten()

        mean_idx = float(np.mean(idx_foliage))
        std_idx  = float(np.std(idx_foliage))
        lo, hi   = self._meta["range"]
        health   = float(np.clip((mean_idx - lo) / (hi - lo), 0.0, 1.0)) * 100

        # Distribuição por zona (sobre pixels de folhagem)
        total_foliage = len(norm_foliage)
        zone_pct      = []
        thresholds    = self.ZONE_THRESHOLDS
        for i in range(len(self.ZONE_LABELS)):
            if i == len(self.ZONE_LABELS) - 1:
                mask = norm_foliage >= thresholds[i]
            else:
                mask = (norm_foliage >= thresholds[i]) & \
                       (norm_foliage <  thresholds[i + 1])
            zone_pct.append(float(np.sum(mask)) / max(total_foliage, 1) * 100)

        green_coverage = zone_pct[3] + zone_pct[4]  # Bom + Ótimo

        # Rótulos de saúde
        if health < 20:   health_label = "Crítico"
        elif health < 40: health_label = "Ruim"
        elif health < 60: health_label = "Moderado"
        elif health < 80: health_label = "Bom"
        else:             health_label = "Ótimo"

        uniformity_label = "Uniforme"   if std_idx < 0.15 else \
                           "Moderado"   if std_idx < 0.25 else "Heterogêneo"

        metrics = {
            "mean_index":         round(mean_idx, 6),
            "std_index":          round(std_idx, 6),
            "min_index":          round(float(np.min(idx_foliage)), 6),
            "max_index":          round(float(np.max(idx_foliage)), 6),
            "health_pct":         round(health, 2),
            "health_label":       health_label,
            "classification":     classify_health(health),
            "green_coverage_pct": round(green_coverage, 2),
            "zone_pct":           [round(p, 2) for p in zone_pct],
            "uniformity_label":   uniformity_label,
        }

        # Métricas de segmentação (apenas no modo cultura)
        if seg_masks:
            metrics["seg_foliage_pct"] = round(seg_masks.get("foliage_pct", 0.0), 2)
            metrics["seg_fruit_pct"]   = round(seg_masks.get("fruit_pct",   0.0), 2)
            metrics["seg_soil_pct"]    = round(seg_masks.get("soil_pct",    0.0), 2)

        return metrics

    def _diagnose(self, m: dict) -> list:
        items = []
        h     = m["health_pct"]
        crop  = self.crop

        # ── Saúde geral da folhagem ───────────────────────────────────────────
        if h >= 70:
            items.append({"level": "ok",
                          "message": f"Folhagem em condição saudável ({h:.1f}%)."})
        elif h >= 45:
            items.append({"level": "warning",
                          "message": f"Saúde moderada ({h:.1f}%) — monitoramento recomendado."})
        else:
            items.append({"level": "critical",
                          "message": f"Folhagem em condição crítica ({h:.1f}%) — intervenção necessária."})

        # ── Zona crítica expressiva ───────────────────────────────────────────
        crit_pct = m["zone_pct"][0]
        if crit_pct > 15:
            items.append({"level": "critical",
                          "message": f"{crit_pct:.1f}% da folhagem em zona crítica — "
                                     "verificar irrigação, pragas e nutrição."})

        # ── Alta cobertura ótima ──────────────────────────────────────────────
        opt_pct = m["zone_pct"][4]
        if opt_pct > 40:
            items.append({"level": "ok",
                          "message": f"{opt_pct:.1f}% da folhagem em zona ótima de clorofila."})

        # ── Uniformidade ─────────────────────────────────────────────────────
        if m["std_index"] > 0.25:
            items.append({"level": "warning",
                          "message": f"Alta variabilidade espacial (σ={m['std_index']:.3f}) — "
                                     "possível foco de infestação ou deficiência localizada."})
        else:
            items.append({"level": "info",
                          "message": f"Distribuição uniforme (σ={m['std_index']:.3f}) — "
                                     "desenvolvimento regular da folhagem."})

        # ── Diagnósticos específicos para tomate ──────────────────────────────
        if crop == "tomato" and "seg_foliage_pct" in m:
            fol_pct   = m["seg_foliage_pct"]
            fruit_pct = m["seg_fruit_pct"]
            soil_pct  = m["seg_soil_pct"]

            if fol_pct < 15:
                items.append({"level": "critical",
                              "message": f"Cobertura foliar muito baixa ({fol_pct:.1f}%) — "
                                         "possível desfolhamento ou doença severa."})
            elif fol_pct < 30:
                items.append({"level": "warning",
                              "message": f"Cobertura foliar moderada ({fol_pct:.1f}%) — "
                                         "monitorar crescimento vegetativo."})
            else:
                items.append({"level": "ok",
                              "message": f"Boa cobertura foliar detectada ({fol_pct:.1f}% da imagem)."})

            if fruit_pct > 25:
                items.append({"level": "ok",
                              "message": f"Alta carga de frutos ({fruit_pct:.1f}%) — "
                                         "produtividade elevada. Atenção à nutrição."})
            elif fruit_pct > 10:
                items.append({"level": "info",
                              "message": f"Carga de frutos moderada ({fruit_pct:.1f}%)."})

            if soil_pct > 40:
                items.append({"level": "info",
                              "message": f"Solo/substrato visível em {soil_pct:.1f}% da imagem — "
                                         "câmera posicionada próxima ao corredor."})

        elif m.get("green_coverage_pct", 100) < 30:
            items.append({"level": "critical",
                          "message": "Baixa cobertura vegetal (<30%) — possível deficiência "
                                     "nutricional ou estresse hídrico severo."})

        return items

    # ------------------------------------------------------------------
    # Figura matplotlib completa
    # ------------------------------------------------------------------

    def _build_figure(self) -> plt.Figure:
        r = self._results
        m = r["metrics"]

        fig = plt.figure(figsize=(16, 10), facecolor="#1C1C1E")
        gs  = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

        TEXT  = "#F0F0F0"
        MUTED = "#A0A0A0"

        def style_ax(ax, title=""):
            ax.set_facecolor("#2C2C2E")
            for sp in ax.spines.values():
                sp.set_color("#444")
            ax.tick_params(colors=MUTED, labelsize=8)
            if title:
                ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=6)

        # --- Imagem original ---
        ax0 = fig.add_subplot(gs[0:2, 0])
        ax0.imshow(cv2.cvtColor(r["original_bgr"], cv2.COLOR_BGR2RGB))
        style_ax(ax0, "Imagem original")
        ax0.axis("off")

        # --- Heatmap ---
        ax1 = fig.add_subplot(gs[0:2, 1])
        heat_rgb = cv2.cvtColor(r["heatmap_bgr"], cv2.COLOR_BGR2RGB)
        im = ax1.imshow(heat_rgb, vmin=0, vmax=1)
        style_ax(ax1, f"Mapa de saúde — {self.index_name}")
        ax1.axis("off")
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                cmap=HEALTH_CMAP,
                norm=mcolors.Normalize(vmin=0, vmax=1),
            ),
            ax=ax1, fraction=0.046, pad=0.04,
        )
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["Crítico", "Ruim", "Moderado", "Bom", "Ótimo"])
        cbar.ax.yaxis.set_tick_params(colors=MUTED, labelsize=7)
        cbar.outline.set_edgecolor("#444")

        # --- Mapa de zonas ---
        ax2 = fig.add_subplot(gs[0:2, 2])
        ax2.imshow(cv2.cvtColor(r["zones_bgr"], cv2.COLOR_BGR2RGB))
        style_ax(ax2, "Mapa de zonas")
        ax2.axis("off")

        # --- Índice bruto OU Mapa de Segmentação HSV ---
        ax3 = fig.add_subplot(gs[0:2, 3])
        if r.get("segment_bgr") is not None:
            ax3.imshow(cv2.cvtColor(r["segment_bgr"], cv2.COLOR_BGR2RGB))
            style_ax(ax3, "Segmentação HSV")
            ax3.axis("off")
            # Legenda da segmentação
            legend_patches = [
                plt.Rectangle((0,0), 1, 1, color=(34/255, 139/255, 34/255),  label="Folhagem"),
                plt.Rectangle((0,0), 1, 1, color=(230/255,100/255, 0/255),   label="Frutos"),
                plt.Rectangle((0,0), 1, 1, color=(101/255, 82/255, 45/255),  label="Solo"),
                plt.Rectangle((0,0), 1, 1, color=(100/255,100/255,100/255),  label="Outros"),
            ]
            ax3.legend(handles=legend_patches, loc="lower right",
                       fontsize=6, labelcolor=TEXT, facecolor="#2C2C2E",
                       edgecolor="#444", framealpha=0.85)
            # Mostra % de cada segmento
            if "seg_foliage_pct" in m:
                seg_txt = (
                    f"Folhagem : {m['seg_foliage_pct']:.1f}%\n"
                    f"Frutos   : {m['seg_fruit_pct']:.1f}%\n"
                    f"Solo     : {m['seg_soil_pct']:.1f}%"
                )
                ax3.text(0.02, 0.02, seg_txt, transform=ax3.transAxes,
                         va="bottom", ha="left", color=TEXT, fontsize=7,
                         bbox=dict(facecolor="#2C2C2E", edgecolor="#555",
                                   boxstyle="round,pad=0.4"))
        else:
            idx_disp = ax3.imshow(r["index_norm"], cmap="RdYlGn", vmin=0, vmax=1)
            style_ax(ax3, f"Índice {self.index_name} (raw)")
            ax3.axis("off")
            cbar2 = plt.colorbar(idx_disp, ax=ax3, fraction=0.046, pad=0.04)
            cbar2.ax.yaxis.set_tick_params(colors=MUTED, labelsize=7)
            cbar2.outline.set_edgecolor("#444")

        # --- Distribuição por zona (barras horizontais) ---
        ax4 = fig.add_subplot(gs[2, 0:2])
        style_ax(ax4, "Distribuição por zona (%)")
        y_pos = range(len(self.ZONE_LABELS))
        bars  = ax4.barh(
            list(y_pos),
            m["zone_pct"],
            color=self.ZONE_COLORS,
            height=0.6,
        )
        ax4.set_yticks(list(y_pos))
        ax4.set_yticklabels(self.ZONE_LABELS, color=TEXT, fontsize=8)
        ax4.set_xlabel("Porcentagem de pixels (%)", color=MUTED, fontsize=8)
        ax4.set_xlim(0, 105)
        for bar, val in zip(bars, m["zone_pct"]):
            ax4.text(
                val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center", ha="left", color=TEXT, fontsize=8,
            )
        ax4.tick_params(axis="x", colors=MUTED)

        # --- Histograma do índice ---
        ax5 = fig.add_subplot(gs[2, 2])
        style_ax(ax5, f"Histograma — {self.index_name}")
        flat = r["index_map"].flatten()
        n, bins, patches = ax5.hist(flat, bins=60, color="#4A90D9", alpha=0.85)
        for patch, left in zip(patches, bins[:-1]):
            lo, hi = self._meta["range"]
            norm   = (left - lo) / (hi - lo)
            patch.set_facecolor(HEALTH_CMAP(np.clip(norm, 0, 1)))
        ax5.axvline(m["mean_index"], color="#FFD700", lw=1.5, linestyle="--",
                    label=f"Média: {m['mean_index']:+.3f}")
        ax5.legend(fontsize=7, labelcolor=TEXT, facecolor="#2C2C2E",
                   edgecolor="#444")
        ax5.set_xlabel(self.index_name, color=MUTED, fontsize=8)
        ax5.set_ylabel("Frequência", color=MUTED, fontsize=8)

        # --- Painel de métricas e diagnóstico ---
        ax6 = fig.add_subplot(gs[2, 3])
        ax6.axis("off")
        style_ax(ax6, "Diagnóstico")

        summary = (
            f"Saúde: {m['health_pct']:.1f}% → {m['health_label']}\n"
            f"Cobertura vegetal: {m['green_coverage_pct']:.1f}%\n"
            f"Média {self.index_name}: {m['mean_index']:+.4f}\n"
            f"Desvio padrão: {m['std_index']:.4f}\n"
            f"Uniformidade: {m['uniformity_label']}\n\n"
        )
        for item in r["diagnosis"]:
            icon = {"ok": "✓", "warning": "⚠", "critical": "✗", "info": "→"}.get(
                item["level"], "·"
            )
            summary += f"{icon} {item['message']}\n"

        ax6.text(
            0.02, 0.98, summary,
            transform=ax6.transAxes,
            va="top", ha="left",
            color=TEXT, fontsize=7,
            wrap=True,
            bbox=dict(facecolor="#2C2C2E", edgecolor="#444",
                      boxstyle="round,pad=0.5"),
        )

        # Título principal
        fig.suptitle(
            f"Sistema Inteligente de Monitoramento Agrícola  ·  Módulo 3\n"
            f"Índice: {self.index_name} — {self._meta['description']}  "
            f"({self._meta['reference']})",
            color=TEXT, fontsize=11, fontweight="bold", y=1.01,
        )

        return fig


# ---------------------------------------------------------------------------
# Modo pasta — processa todas as imagens de um diretório
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def run_folder(
    folder_path:  str,
    index_name:   str  = "VARI",
    output_dir:   str  = "resultados",
    save_reports: bool = True,
    no_show:      bool = True,
    crop:         str  = None,
):
    """
    Processa em lote todas as imagens de uma pasta.

    Parâmetros
    ----------
    folder_path  : caminho para a pasta com as imagens
    index_name   : índice de vegetação a utilizar
    output_dir   : pasta onde salvar relatórios e log.txt
    save_reports : se True salva heatmap + figura por imagem
    no_show      : se True não abre janelas gráficas
    crop         : tipo de cultura ('tomato' ou None para genérico)
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Pasta não encontrada: {folder_path}")

    images = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    if not images:
        print(f"[Módulo 3] Nenhuma imagem encontrada em: {folder_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.txt")
    logger   = HealthLogger(log_path=log_path, index_name=index_name)
    analyzer = VegetationAnalyzer(index_name=index_name, crop=crop)

    counts = {"saudavel": 0, "alerta": 0, "critico": 0, "erro": 0}
    total  = len(images)

    print(f"\n[Módulo 3] Processando {total} imagem(ns) em: {folder_path}")
    print(f"           Índice: {index_name}  |  Cultura: {crop or 'generico'}  |  Saída: {output_dir}")
    print(f"           Log   : {log_path}")
    print("-" * 60)

    for i, filename in enumerate(images, start=1):
        img_path = os.path.join(folder_path, filename)
        print(f"  [{i:>3}/{total}] {filename} ... ", end="", flush=True)

        try:
            results        = analyzer.analyze_image(img_path)
            m              = results["metrics"]
            classification = m["classification"]
            counts[classification] += 1

            logger.record(filename, m["health_pct"], classification)
            print(f"{classification.upper():<10}  ({m['health_pct']:.1f}%)")

            if save_reports:
                prefix = os.path.splitext(filename)[0]
                analyzer.save_report(output_dir=output_dir, prefix=prefix)

        except Exception as exc:
            counts["erro"] += 1
            logger.record(filename, 0.0, "erro")
            print(f"ERRO: {exc}")

    logger.summary(counts)

    print("-" * 60)
    print(f"  Concluído. Resultados:")
    print(f"    Saudavel : {counts['saudavel']}")
    print(f"    Alerta   : {counts['alerta']}")
    print(f"    Critico  : {counts['critico']}")
    if counts["erro"]:
        print(f"    Erros    : {counts['erro']}")
    print(f"  Log salvo em: {log_path}\n")


# ---------------------------------------------------------------------------
# Modo câmera ao vivo
# ---------------------------------------------------------------------------

def run_live_camera(
    index_name: str = "VARI",
    cam_id:     int = 0,
    output_dir: str = "resultados_camera",
    crop:       str = None,
):
    """
    Executa análise em tempo real via câmera (webcam ou RTSP).

    Teclas:
        [q] — sair
        [s] — salvar frame atual (relatório + entrada no log)
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.txt")
    logger   = HealthLogger(log_path=log_path, index_name=index_name)
    analyzer = VegetationAnalyzer(index_name=index_name, max_dim=640, crop=crop)
    cap      = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera: {cam_id}")

    print(f"\n[Módulo 3] Câmera ao vivo — índice {index_name}  |  cultura: {crop or 'generico'}")
    print(f"           Log: {log_path}")
    print("  Teclas: [q] sair  |  [s] salvar frame e registrar no log\n")

    frame_count  = 0
    saved_count  = 0
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Processa 1 em cada 5 frames (economiza CPU no Raspberry Pi)
        if frame_count % 5 == 0:
            last_results   = analyzer.analyze_frame(frame)
            m              = last_results["metrics"]
            classification = m["classification"]

            color_map = {
                "saudavel": (0, 200, 80),
                "alerta":   (0, 200, 230),
                "critico":  (0, 60, 220),
            }
            color   = color_map.get(classification, (255, 255, 255))
            heatmap = last_results["heatmap_bgr"].copy()

            # Escolhe segunda janela: segmentação (tomato) ou heatmap puro
            if crop == "tomato" and last_results.get("segment_bgr") is not None:
                second_panel = last_results["segment_bgr"].copy()
                panel_label  = "Segmentacao HSV"
            else:
                second_panel = heatmap
                panel_label  = "Mapa de Saude"

            overlay_lines = [
                f"Indice  : {index_name}",
                f"Cultura : {crop or 'generico'}",
                f"Saude   : {m['health_pct']:.1f}%  ({m['health_label']})",
                f"Status  : {classification.upper()}",
                f"[s] salvar  [q] sair",
            ]
            if "seg_foliage_pct" in m:
                overlay_lines.insert(4,
                    f"Folhagem: {m['seg_foliage_pct']:.0f}%  "
                    f"Frutos: {m['seg_fruit_pct']:.0f}%")

            for j, txt in enumerate(overlay_lines):
                is_status = ("Status" in txt)
                cv2.putText(
                    heatmap, txt,
                    (10, 22 + j * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    color if is_status else (255, 255, 255),
                    2 if is_status else 1,
                    cv2.LINE_AA,
                )

            combined = np.hstack([
                cv2.resize(frame,        (480, 360)),
                cv2.resize(heatmap,      (480, 360)),
                cv2.resize(second_panel, (480, 360)),
            ])
            cv2.imshow(f"Modulo 3 — Original | Heatmap | {panel_label}", combined)

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s") and last_results is not None:
            saved_count += 1
            m              = last_results["metrics"]
            classification = m["classification"]
            frame_name     = f"camera_frame_{saved_count:04d}.png"

            prefix = f"camera_frame_{saved_count:04d}"
            analyzer.save_report(output_dir=output_dir, prefix=prefix)
            analyzer.print_summary()

            logger.record(frame_name, m["health_pct"], classification)
            print(f"  [LOG] {frame_name} - {classification}  ({m['health_pct']:.1f}%)")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Módulo 3] Sessão encerrada. {saved_count} frame(s) salvo(s).")
    print(f"           Log: {log_path}\n")


# ---------------------------------------------------------------------------
# Funções auxiliares do menu interativo
# ---------------------------------------------------------------------------

def _menu_escolher_indice(crop: str = None) -> str:
    """Exibe submenu para escolha do índice de vegetação."""
    indices = list(INDICES.keys())
    print("\n  Índices disponíveis:")
    for i, nome in enumerate(indices, 1):
        meta = INDICES[nome]
        desc = meta["description"]
        ref  = meta["reference"]
        nota = meta.get("crop_note", "")
        rec  = " ← RECOMENDADO" if (crop and VegetationAnalyzer.CROP_RECOMMENDED.get(crop) == nome) else ""
        print(f"    [{i}] {nome:<10} — {desc} ({ref}){rec}")
        if nota:
            print(f"          {nota}")
    print()

    default = VegetationAnalyzer.CROP_RECOMMENDED.get(crop, "VARI") if crop else "VARI"
    while True:
        escolha = input(f"  Escolha o índice [1-{len(indices)}] (Enter = {default}): ").strip()
        if escolha == "":
            return default
        if escolha.isdigit() and 1 <= int(escolha) <= len(indices):
            return indices[int(escolha) - 1]
        print("  ⚠  Opção inválida. Tente novamente.")


def _menu_escolher_cultura() -> str:
    """Solicita o tipo de cultura para ativar segmentação específica."""
    culturas = {
        "0": None,
        "1": "tomato",
    }
    print("\n  Tipo de cultura:")
    print("    [0]  Genérico        — vegetação verde em geral")
    print("    [1]  Tomate          — segmenta folhagem, frutos e solo via HSV")
    print()
    while True:
        escolha = input("  Escolha [0-1] (Enter = 0): ").strip()
        if escolha in culturas:
            return culturas[escolha]
        if escolha == "":
            return None
        print("  ⚠  Opção inválida.")


def _menu_escolher_saida(padrao: str = "resultados") -> str:
    """Solicita o diretório de saída."""
    resp = input(f"  Diretório de saída [Enter = '{padrao}']: ").strip()
    return resp if resp else padrao


def _executar_imagem_unica(no_show: bool = False):
    """Fluxo interativo para análise de imagem única."""
    print("\n" + "─" * 56)
    print("  MODO: Imagem única")
    print("─" * 56)

    while True:
        caminho = input("  Caminho da imagem: ").strip().strip('"').strip("'")
        if os.path.isfile(caminho):
            break
        print(f"  ⚠  Arquivo não encontrado: '{caminho}'. Tente novamente.")

    crop       = _menu_escolher_cultura()
    index_name = _menu_escolher_indice(crop=crop)
    output_dir = _menu_escolher_saida()

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.txt")
    logger   = HealthLogger(log_path=log_path, index_name=index_name)

    analyzer = VegetationAnalyzer(index_name=index_name, crop=crop)
    results  = analyzer.analyze_image(caminho)
    m        = results["metrics"]

    analyzer.print_summary()
    analyzer.save_report(output_dir=output_dir)

    filename = os.path.basename(caminho)
    logger.record(filename, m["health_pct"], m["classification"])
    print(f"[LOG] {filename} - {m['classification']}  ({m['health_pct']:.1f}%)")
    print(f"      Registrado em: {log_path}")

    if not no_show:
        resp = input("\n  Exibir relatório gráfico? [S/n]: ").strip().lower()
        if resp in ("", "s", "sim", "y", "yes"):
            analyzer.show()


def _executar_pasta():
    """Fluxo interativo para processamento em lote."""
    print("\n" + "─" * 56)
    print("  MODO: Pasta (lote)")
    print("─" * 56)

    while True:
        caminho = input("  Caminho da pasta: ").strip().strip('"').strip("'")
        if os.path.isdir(caminho):
            break
        print(f"  ⚠  Pasta não encontrada: '{caminho}'. Tente novamente.")

    crop       = _menu_escolher_cultura()
    index_name = _menu_escolher_indice(crop=crop)
    output_dir = _menu_escolher_saida()

    resp_rel     = input("  Salvar relatório individual por imagem? [S/n]: ").strip().lower()
    save_reports = resp_rel not in ("n", "nao", "não", "no")

    run_folder(
        folder_path=caminho,
        index_name=index_name,
        output_dir=output_dir,
        save_reports=save_reports,
        no_show=True,
        crop=crop,
    )


def _executar_camera():
    """Fluxo interativo para câmera ao vivo."""
    print("\n" + "─" * 56)
    print("  MODO: Câmera ao vivo")
    print("─" * 56)

    cam_input = input("  ID ou URL da câmera [Enter = 0]: ").strip()
    if cam_input == "":
        cam_id = 0
    elif cam_input.isdigit():
        cam_id = int(cam_input)
    else:
        cam_id = cam_input

    crop       = _menu_escolher_cultura()
    index_name = _menu_escolher_indice(crop=crop)
    output_dir = _menu_escolher_saida("resultados_camera")

    run_live_camera(
        index_name=index_name,
        cam_id=cam_id,
        output_dir=output_dir,
        crop=crop,
    )


def menu_interativo():
    """
    Exibe o menu principal e executa o modo escolhido pelo usuário.
    Repete até o usuário optar por sair.
    """
    banner = r"""
  ╔══════════════════════════════════════════════════════╗
  ║   Módulo 3 — Sistema Inteligente de Monitoramento   ║
  ║              Agrícola  ·  FHO 2025                  ║
  ╚══════════════════════════════════════════════════════╝
    """
    while True:
        print(banner)
        print("  Selecione o modo de execução:")
        print()
        print("    [1]  Imagem única      — analisa uma foto")
        print("    [2]  Pasta (lote)      — processa todas as imagens de uma pasta")
        print("    [3]  Câmera ao vivo    — análise em tempo real via webcam/RTSP")
        print()
        print("    [0]  Sair")
        print()

        opcao = input("  Opção: ").strip()

        if opcao == "1":
            _executar_imagem_unica()
        elif opcao == "2":
            _executar_pasta()
        elif opcao == "3":
            _executar_camera()
        elif opcao == "0":
            print("\n  Encerrando. Até mais!\n")
            break
        else:
            print("\n  ⚠  Opção inválida. Digite 1, 2, 3 ou 0.\n")
            continue

        # Após cada execução, pergunta se deseja continuar
        print()
        continuar = input("  Voltar ao menu principal? [S/n]: ").strip().lower()
        if continuar in ("n", "nao", "não", "no"):
            print("\n  Encerrando. Até mais!\n")
            break


# ---------------------------------------------------------------------------
# CLI  (mantida para uso via terminal com argumentos)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Módulo 3 — Análise de Saúde da Vegetação (RGB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python vegetation_analysis.py --image plantacao.jpg
  python vegetation_analysis.py --image tomate.jpg --index ExGR --crop tomato
  python vegetation_analysis.py --folder imagens/ --index ExGR --crop tomato
  python vegetation_analysis.py --folder imagens/ --no-reports --crop tomato
  python vegetation_analysis.py --camera 0 --index TomatoVI --crop tomato
  python vegetation_analysis.py --camera rtsp://192.168.1.100/stream --crop tomato

  Sem argumentos → abre o menu interativo.
        """,
    )

    # ── Fonte de entrada (mutuamente exclusivos) ──────────────────────────────
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image",  type=str, help="Caminho para uma imagem")
    src.add_argument("--folder", type=str, help="Pasta com imagens para processamento em lote")
    src.add_argument("--camera", type=str, help="ID ou URL da câmera (ex: 0, rtsp://...)")

    # ── Opções gerais ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--index", type=str, default=None,
        choices=list(INDICES.keys()),
        help="Índice de vegetação (padrão: ExGR para tomato, VARI para genérico)",
    )
    parser.add_argument(
        "--crop", type=str, default=None,
        choices=["tomato"],
        help="Tipo de cultura para segmentação HSV específica (ex: tomato)",
    )
    parser.add_argument(
        "--output", type=str, default="resultados",
        help="Diretório de saída para relatórios e log.txt (padrão: resultados/)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Não exibe janela gráfica (útil em servidores/Raspberry Pi sem display)",
    )
    parser.add_argument(
        "--no-reports", action="store_true",
        help="No modo --folder, não salva relatório individual por imagem",
    )

    args = parser.parse_args()

    # ── Resolve índice padrão baseado na cultura ──────────────────────────────
    if args.index is None:
        args.index = VegetationAnalyzer.CROP_RECOMMENDED.get(args.crop, "VARI")

    # ── Nenhum argumento de entrada → menu interativo ─────────────────────────
    if args.image is None and args.folder is None and args.camera is None:
        menu_interativo()
        return

    # ── Câmera ────────────────────────────────────────────────────────────────
    if args.camera is not None:
        cam_id = int(args.camera) if args.camera.isdigit() else args.camera
        run_live_camera(
            index_name=args.index,
            cam_id=cam_id,
            output_dir=args.output,
            crop=args.crop,
        )
        return

    # ── Pasta ─────────────────────────────────────────────────────────────────
    if args.folder is not None:
        run_folder(
            folder_path=args.folder,
            index_name=args.index,
            output_dir=args.output,
            save_reports=not args.no_reports,
            no_show=args.no_show,
            crop=args.crop,
        )
        return

    # ── Imagem única ──────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, "log.txt")
    logger   = HealthLogger(log_path=log_path, index_name=args.index)

    analyzer = VegetationAnalyzer(index_name=args.index, crop=args.crop)
    results  = analyzer.analyze_image(args.image)
    m        = results["metrics"]

    analyzer.print_summary()
    analyzer.save_report(output_dir=args.output)

    filename = os.path.basename(args.image)
    logger.record(filename, m["health_pct"], m["classification"])
    print(f"[LOG] {filename} - {m['classification']}  ({m['health_pct']:.1f}%)")
    print(f"      Registrado em: {log_path}")

    if not args.no_show:
        analyzer.show()


if __name__ == "__main__":
    main()