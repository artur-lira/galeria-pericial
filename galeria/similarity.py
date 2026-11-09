"""Detecção de imagens semelhantes.

Camadas de análise (da mais rápida para a mais robusta):

1. SHA-256        -> cópia bit a bit idêntica.
2. pHash (DCT)    -> mesma imagem redimensionada, recomprimida, miniatura/cache,
                     pequenas alterações de cor/brilho. Calculado também para as
                     8 transformações diedrais (rotações 90/180/270 + espelhamentos).
3. ORB + RANSAC   -> recortes (crop), imagem contida em outra, imagens com
                     sobreposições/edições locais, mudança de escala e rotação
                     arbitrária. A homografia estimada permite localizar o
                     recorte dentro da imagem maior e gerar mapa de diferenças.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

NORM_MAX = 900          # lado máximo para extração de pontos-chave
ORB_FEATURES = 1500
PHASH_NEAR = 10         # distância de Hamming (0-64) para "mesma imagem"
PHASH_SIMILAR = 16      # semelhança visual fraca (só reportada se ORB confirmar)
MIN_INLIERS = 15


# ---- Hashes perceptuais -------------------------------------------------
def _phash_gray(gray32: np.ndarray) -> int:
    dct = cv2.dct(gray32.astype(np.float32))
    low = dct[:8, :8].flatten()
    med = np.median(low[1:])
    bits = low > med
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def dihedral_phashes(img: Image.Image) -> list[int]:
    """pHash das 8 transformações: [id, rot90, rot180, rot270, flipH, flipH+90, flipH+180, flipH+270]."""
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
    out = []
    for flip in (False, True):
        base = np.fliplr(g) if flip else g
        for k in range(4):
            out.append(_phash_gray(np.ascontiguousarray(np.rot90(base, k))))
    return out


def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count("1")


TRANSFORM_NAMES = ["", "rotacionada 90°", "rotacionada 180°", "rotacionada 270°",
                   "espelhada", "espelhada + 90°", "espelhada + 180°", "espelhada + 270°"]


# ---- Uniformidade (imagens/regiões sem conteúdo) ------------------------
FLAT_TOL = 10        # tolerância de nível de cinza em torno da mediana
FLAT_FRAC = 0.97     # fração mínima de pixels "iguais" para considerar a região uniforme
COLOR_TOL = 60       # diferença máxima de cor média (0-255) para aceitar "mesma imagem" por pHash


def region_flatmask(img: Image.Image) -> tuple[int, str]:
    """Bitmask (15 bits, mesma ordem dos vetores da CNN) das regiões uniformes + cor média 'r,g,b'.

    Região uniforme = >= 97% dos pixels a até ±10 níveis da mediana (fundo branco, preto,
    cor sólida, céu liso...). Esses trechos não carregam informação para comparação e
    geravam falsos positivos (ex.: imagem toda branca x imagem com fundo branco).
    """
    from .engine import _REGIONS  # import leve (não carrega o PyTorch)
    rgb = img.convert("RGB")
    if max(rgb.size) > 256:
        rgb = rgb.copy()
        rgb.thumbnail((256, 256))
    a = np.asarray(rgb, dtype=np.int16)
    g = np.asarray(rgb.convert("L"), dtype=np.int16)
    H, W = g.shape
    regs = [_REGIONS[0], _REGIONS[0]] + list(_REGIONS[1:])
    mask = 0
    for k, (x0, y0, x1, y1) in enumerate(regs):
        r0, c0 = int(y0 * H), int(x0 * W)
        sub = g[r0:max(r0 + 1, int(round(y1 * H))), c0:max(c0 + 1, int(round(x1 * W)))]
        if sub.size == 0:
            continue
        med = np.median(sub)
        if (np.abs(sub - med) <= FLAT_TOL).mean() >= FLAT_FRAC:
            mask |= 1 << k
    mean = a.reshape(-1, 3).mean(0)
    return mask, ",".join(str(int(round(v))) for v in mean)


def color_distance(a: str | None, b: str | None) -> int:
    try:
        x = [int(v) for v in a.split(",")]
        y = [int(v) for v in b.split(",")]
        return max(abs(i - j) for i, j in zip(x, y))
    except Exception:  # noqa: BLE001
        return 0


# ---- Pontos-chave -------------------------------------------------------
def normalize_gray(img: Image.Image) -> tuple[np.ndarray, float]:
    """Converte para cinza com lado máximo NORM_MAX. Retorna (gray, escala)."""
    w, h = img.size
    s = min(1.0, NORM_MAX / max(w, h)) if max(w, h) > 0 else 1.0
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return np.asarray(img.convert("L")), s


def normalize_rgb(img: Image.Image) -> np.ndarray:
    w, h = img.size
    s = min(1.0, NORM_MAX / max(w, h)) if max(w, h) > 0 else 1.0
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return np.asarray(img.convert("RGB"))


def orb_features(img: Image.Image):
    """Retorna (kp_xy float32 Nx2, desc uint8 Nx32, norm_w, norm_h)."""
    gray, _ = normalize_gray(img)
    # pequenas imagens (miniaturas) são ampliadas para gerar pontos suficientes
    h, w = gray.shape
    up = 1.0
    if max(w, h) < 300:
        up = 300 / max(w, h)
        gray = cv2.resize(gray, (round(w * up), round(h * up)), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=10,
                         edgeThreshold=15, patchSize=31, fastThreshold=10)
    kps, desc = orb.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8), w, h
    xy = np.array([k.pt for k in kps], dtype=np.float32) / up
    return xy, desc, w, h


@dataclass
class Feat:
    xy: np.ndarray
    desc: np.ndarray
    w: int
    h: int


_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _is_convex(p: np.ndarray) -> bool:
    return bool(cv2.isContourConvex(p.reshape(-1, 1, 2).astype(np.float32)))


def match_features(a: Feat, b: Feat):
    """Casa pontos de A em B. Retorna dict ou None.

    dict: inliers, H (A->B, coords normalizadas), frac_a_in_b (área de A
    projetada em B / área de B), frac_b_in_a, poly_a_in_b, poly_b_in_a.
    """
    if len(a.desc) < 8 or len(b.desc) < 8:
        return None
    knn = _matcher.knnMatch(a.desc, b.desc, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.78 * m[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    src = a.xy[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = b.xy[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0, maxIters=3000, confidence=0.995)
    if H is None or mask is None:
        return None
    inl = int(mask.sum())
    if inl < MIN_INLIERS or inl < 0.25 * len(good):
        return None
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    ca = np.float32([[0, 0], [a.w, 0], [a.w, a.h], [0, a.h]]).reshape(-1, 1, 2)
    cb = np.float32([[0, 0], [b.w, 0], [b.w, b.h], [0, b.h]]).reshape(-1, 1, 2)
    pa = cv2.perspectiveTransform(ca, H).reshape(-1, 2)
    pb = cv2.perspectiveTransform(cb, Hinv).reshape(-1, 2)
    if not (_is_convex(pa) and _is_convex(pb)):
        return None
    area_a_in_b = _poly_area(pa)
    area_b_in_a = _poly_area(pb)
    if area_a_in_b < 1 or area_b_in_a < 1:
        return None
    # área de interseção (parte de A visível em B)
    rect_b = np.float32([[0, 0], [b.w, 0], [b.w, b.h], [0, b.h]])
    inter_ab, _ = cv2.intersectConvexConvex(pa.astype(np.float32), rect_b)
    rect_a = np.float32([[0, 0], [a.w, 0], [a.w, a.h], [0, a.h]])
    inter_ba, _ = cv2.intersectConvexConvex(pb.astype(np.float32), rect_a)
    return {
        "inliers": inl,
        "good": len(good),
        "H": H,
        "frac_a_in_b": float(inter_ab / (b.w * b.h)),   # quanto de B é coberto por A
        "frac_b_in_a": float(inter_ba / (a.w * a.h)),   # quanto de A é coberto por B
        "poly_a_in_b": pa,
        "poly_b_in_a": pb,
    }


# ---- Classificação ------------------------------------------------------
CAT_IDENTICAL = "Cópia idêntica (SHA-256)"
CAT_NEAR = "Mesma imagem (redimensionada/recomprimida)"
CAT_THUMB = "Miniatura / cache reduzido"
CAT_TRANSFORM = "Mesma imagem rotacionada/espelhada"
CAT_CROP_OF = "Esta imagem é RECORTE da outra"
CAT_CONTAINS = "A outra é RECORTE desta imagem"
CAT_EDITED = "Mesma cena/imagem com alterações"
CAT_OVERLAP = "Conteúdo parcialmente coincidente"

CAT_ORDER = [CAT_IDENTICAL, CAT_NEAR, CAT_THUMB, CAT_TRANSFORM, CAT_CROP_OF,
             CAT_CONTAINS, CAT_EDITED, CAT_OVERLAP]


def classify(q: dict, c: dict, qf: Feat | None, cf: Feat | None, use_orb: bool = True):
    """Compara registro q (consulta) com c (candidato). Retorna dict ou None."""
    if q["sha256"] and q["sha256"] == c["sha256"]:
        return {"cat": CAT_IDENTICAL, "score": 100.0, "detail": "Hash SHA-256 coincidente"}
    qh, ch = q["phash"], c["phash"]
    d_id = hamming(qh[0], ch[0])
    d_best, t_best = min((hamming(qh[0], ch[k]), k) for k in range(8))
    qa, ca = q["width"] * q["height"], c["width"] * c["height"]
    if d_id <= PHASH_NEAR:
        ratio = ca / qa if qa else 1
        score = 100 - d_id * 2.5
        dim = f"{c['width']}x{c['height']} vs {q['width']}x{q['height']}"
        if ratio < 0.35:
            return {"cat": CAT_THUMB, "score": score, "detail": f"pHash dist={d_id}; {dim}"}
        return {"cat": CAT_NEAR, "score": score, "detail": f"pHash dist={d_id}; {dim}"}
    if d_best <= PHASH_NEAR - 2 and t_best != 0:
        return {"cat": CAT_TRANSFORM, "score": 100 - d_best * 2.5,
                "detail": f"{TRANSFORM_NAMES[t_best]}; pHash dist={d_best}"}
    if not use_orb or qf is None or cf is None:
        return None
    return classify_geometry(qf, cf, d_id)


SAME_FRAME = 0.95   # cobertura mínima (nos dois sentidos) para "mesmo enquadramento"
IN_FRAME = 0.85     # cobertura mínima da imagem menor dentro da maior para ser recorte


def classify_geometry(qf: Feat, cf: Feat, d_id: int | None = None, m: dict | None = None):
    """Relação geométrica entre Q e C pelos pontos ORB + homografia (vale para recortes de
    QUALQUER proporção, inclusive zoom central com a mesma proporção do original).

    fa = fração de C coberta por Q; fb = fração de Q coberta por C.
      fb >= 0,85 e fa claramente menor  -> Q está contida em C  => Q é RECORTE de C
      fa >= 0,85 e fb claramente menor  -> C está contida em Q  => C é RECORTE de Q
      ambas >= 0,95                     -> mesmo enquadramento (com alterações)
      demais                            -> sobreposição parcial
    """
    m = m or match_features(qf, cf)
    if m is None:
        return None
    inl = m["inliers"]
    conf = min(99.0, 50 + inl * 0.8)
    fa, fb = m["frac_a_in_b"], m["frac_b_in_a"]
    ph = f"; pHash dist={d_id}" if d_id is not None else ""
    if fa >= SAME_FRAME and fb >= SAME_FRAME:
        return {"cat": CAT_EDITED, "score": conf, "match": m,
                "detail": f"{inl} pontos coincidentes; mesmo enquadramento{ph}"}
    # recorte girado/com preenchimento (bordas pretas/brancas reduzem a cobertura medida)
    rot_crop_of = fa < 0.80 and fb >= 0.55 and fb > 1.5 * fa
    rot_contains = fb < 0.80 and fa >= 0.55 and fa > 1.5 * fb
    if (fb >= IN_FRAME and fa < fb - 0.04) or rot_crop_of:
        return {"cat": CAT_CROP_OF, "score": conf, "match": m,
                "detail": f"{inl} pontos coincidentes; ocupa {fa * 100:.0f}% da outra"}
    if (fa >= IN_FRAME and fb < fa - 0.04) or rot_contains:
        return {"cat": CAT_CONTAINS, "score": conf, "match": m,
                "detail": f"{inl} pontos coincidentes; a outra ocupa {fb * 100:.0f}% desta"}
    return {"cat": CAT_OVERLAP, "score": conf * 0.8, "match": m,
            "detail": f"{inl} pontos coincidentes; sobreposição parcial ({fb * 100:.0f}% desta, "
                    f"{fa * 100:.0f}% da outra)"}


def local_edit_percent(a: Image.Image, b: Image.Image) -> float:
    """Para imagens de mesmo enquadramento (pHash próximo): % da área com
    divergência local, após normalizar tamanho, brilho e contraste.
    Recompressão/redimensionamento geram valores próximos de 0."""
    w = min(a.width, b.width, 256)
    h = max(1, round(w * a.height / a.width))
    g1 = np.asarray(a.convert("L").resize((w, h), Image.BILINEAR), np.float32)
    g2 = np.asarray(b.convert("L").resize((w, h), Image.BILINEAR), np.float32)
    g1 = cv2.GaussianBlur(g1, (3, 3), 0)
    g2 = cv2.GaussianBlur(g2, (3, 3), 0)
    k = (g1.std() + 1e-3) / (g2.std() + 1e-3)
    g2 = g2 * k + (g1.mean() - k * g2.mean())
    d = np.abs(g1 - g2) > 40
    d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return float(d.mean() * 100)


def _dihedral(arr: np.ndarray, k: int) -> np.ndarray:
    """Mesma convenção de dihedral_phashes: k<4 rotação k*90°; k>=4 espelha e depois rotaciona."""
    base = np.fliplr(arr) if k >= 4 else arr
    return np.ascontiguousarray(np.rot90(base, k % 4))


CONTENT_REJECT = 35.0   # % do CONTEÚDO divergente acima do qual não é a mesma imagem
CONTENT_EDIT = 3.0      # % do conteúdo divergente acima do qual é "com alterações"


def content_divergence(a: Image.Image, b: Image.Image, k: int = 0) -> tuple[float, float]:
    """Confirma, em resolução de miniatura, se B (transformada por k) mostra o mesmo conteúdo de A.

    Retorna (% do CONTEÚDO divergente, % da ÁREA TOTAL divergente).
    - "Conteúdo" = pixels que não são o fundo (cor da borda) em pelo menos uma das imagens; fundos e
      margens brancas/transparentes não inflam a semelhança (ex.: dois personagens diferentes
      centralizados sobre fundo transparente têm pHash parecido, mas conteúdo diferente).
    - Proporções diferentes (> ~10%) => não é a mesma imagem redimensionada.
    - A comparação de aceitação é feita em 64 px com desfoque, tolerando pequenos deslocamentos
      (capturas de tela, bordas cortadas em poucos pixels).
    """
    ga0 = np.asarray(a.convert("L"), np.float32)
    gb0 = _dihedral(np.asarray(b.convert("L"), np.float32), k)
    ca0 = np.asarray(a.convert("RGB"), np.float32)
    cb0 = _dihedral(np.asarray(b.convert("RGB"), np.float32), k)
    ar_a = ga0.shape[1] / ga0.shape[0]
    ar_b = gb0.shape[1] / gb0.shape[0]
    if abs(np.log(ar_a / ar_b)) > 0.10:
        return 100.0, 100.0

    def prep(w, blur):
        h = max(1, round(w * ga0.shape[0] / ga0.shape[1]))
        x = cv2.resize(ga0, (w, h), interpolation=cv2.INTER_AREA)
        y = cv2.resize(gb0, (w, h), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(x, (blur, blur), 0), cv2.GaussianBlur(y, (blur, blur), 0)

    def bg(g):
        return np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]))

    def diverge(x, y, thr):
        content = (np.abs(x - bg(x)) > 15) | (np.abs(y - bg(y)) > 15)
        if content.mean() < 0.02:
            content[:] = True
        # normalização de brilho/contraste LIMITADA (não pode "achatar" uma imagem na outra)
        kk = float(np.clip((x[content].std() + 1e-3) / (y[content].std() + 1e-3), 0.67, 1.5))
        y2 = y * kk + (x[content].mean() - kk * y[content].mean())
        d = np.abs(x - y2) > thr
        return float((d & content).sum() / max(1, content.sum()) * 100), float(d.mean() * 100)

    pc, _ = diverge(*prep(64, 5), 35)                                   # aceitação (tolerante)
    # alterações locais (fino, em cor): maior diferença entre canais R, G, B
    w2 = int(min(ga0.shape[1], gb0.shape[1], 256))
    h2 = max(1, round(w2 * ga0.shape[0] / ga0.shape[1]))
    x = cv2.GaussianBlur(cv2.resize(ca0, (w2, h2), interpolation=cv2.INTER_AREA), (3, 3), 0)
    y = cv2.GaussianBlur(cv2.resize(cb0, (w2, h2), interpolation=cv2.INTER_AREA), (3, 3), 0)
    y = y + (x.reshape(-1, 3).mean(0) - y.reshape(-1, 3).mean(0))      # compensa só deslocamento de cor
    d = (np.abs(x - y).max(axis=2) > 45).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return pc, float(d.mean() * 100)


def best_transform(q: dict, c: dict) -> int:
    qh, ch = q["phash"], c["phash"]
    return min(range(8), key=lambda k: hamming(qh[0], ch[k]))


# ---- Mapa de diferenças -------------------------------------------------
def difference_map(q_img: Image.Image, c_img: Image.Image, H: np.ndarray | None):
    """Alinha c sobre q (via homografia q->c, ou redimensionamento) e destaca diferenças.

    Retorna (overlay RGB uint8, percentual de pixels divergentes na área comum).
    """
    qa = normalize_rgb(q_img)
    ca = normalize_rgb(c_img)
    hq, wq = qa.shape[:2]
    if H is None:
        warped = cv2.resize(ca, (wq, hq), interpolation=cv2.INTER_AREA)
        mask = np.full((hq, wq), 255, np.uint8)
    else:
        Hinv = np.linalg.inv(H)  # c -> q
        warped = cv2.warpPerspective(ca, Hinv, (wq, hq), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full(ca.shape[:2], 255, np.uint8), Hinv, (wq, hq),
                                   flags=cv2.INTER_NEAREST)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    g1 = cv2.GaussianBlur(cv2.cvtColor(qa, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    # normaliza brilho/contraste para não acusar simples ajuste global
    m = mask > 0
    if m.sum() > 100:
        g2f = g2.astype(np.float32)
        g1f = g1.astype(np.float32)
        a = (g1f[m].std() + 1e-3) / (g2f[m].std() + 1e-3)
        b = g1f[m].mean() - a * g2f[m].mean()
        g2 = np.clip(g2f * a + b, 0, 255).astype(np.uint8)
    diff = cv2.absdiff(g1, g2)
    diff[~m] = 0
    _, th = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    th = cv2.dilate(th, np.ones((5, 5), np.uint8))
    pct = float((th > 0).sum() / max(1, m.sum()) * 100)
    out = qa.copy()
    out[~m] = (out[~m] * 0.35).astype(np.uint8)  # escurece área sem correspondência
    red = np.zeros_like(out)
    red[..., 0] = 255
    sel = th > 0
    out[sel] = (out[sel] * 0.4 + red[sel] * 0.6).astype(np.uint8)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cn in cnts:
        if cv2.contourArea(cn) > 30:
            x, y, w, h = cv2.boundingRect(cn)
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 1)
    return out, pct
