"""Busca de semelhantes de UMA imagem contra TODOS os arquivos da pasta e subpastas,
e análise global (todos x todos).

Fluxo por consulta:
  1. SHA-256 idêntico                         (instantâneo)
  2. pHash das 8 orientações x toda a coleção (vetorizado, milissegundos)
  3. CNN regional x toda a coleção            (GPU/multi-GPU ou CPU, ~0,1-2 s p/ 300 mil)
  4. Verificação geométrica ORB+RANSAC dos candidatos (paralela) -> classificação final
"""
from __future__ import annotations

import io
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

from . import loader, similarity as sim

CNN_TOPK = 60            # candidatos da CNN verificados por consulta
CNN_STRONG = 0.80        # acima disso também entra como candidato (até CNN_MAX)
CNN_MAX = 200
CNN_VISUAL = 0.85        # não confirmado pelo ORB, mas reportado como "visualmente semelhante"
DEFAULT_GLOBAL_THR = 0.70
VISUAL_MAX_PHASH = 22   # "visualmente semelhante" sem ORB exige pHash razoavelmente próximo
BRUTE_FORCE_MAX = 5000   # sem CNN: ORB contra todas até esse tamanho de coleção

CAT_VISUAL = "Visualmente semelhante (não confirmado geometricamente)"
if CAT_VISUAL not in sim.CAT_ORDER:
    sim.CAT_ORDER.append(CAT_VISUAL)


class FeatureCache:
    """Pontos ORB calculados sob demanda (não ficam no disco: 300 mil x 60 KB seria inviável)."""

    def __init__(self, cap=3000):
        self.cap = cap
        self.d: OrderedDict[str, sim.Feat | None] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, path: str) -> sim.Feat | None:
        with self.lock:
            if path in self.d:
                self.d.move_to_end(path)
                return self.d[path]
        try:
            img = loader.to_rgb(loader.open_pil(path, sim.NORM_MAX))
            xy, desc, w, h = sim.orb_features(img)
            f = sim.Feat(xy, desc, w, h)
        except Exception:  # noqa: BLE001
            f = None
        with self.lock:
            self.d[path] = f
            if len(self.d) > self.cap:
                self.d.popitem(last=False)
        return f


FEATS = FeatureCache()


def phash_distances(store, row: int):
    """(dist_identidade, dist_min_8_orientações, índice_transformação) para todas as linhas."""
    P, ok = store.P, store.P_ok
    q0 = P[row, 0]
    d = np.bitwise_count(P ^ q0).astype(np.int16)  # (N,8)
    d[~ok] = 99
    flat = getattr(store, "flat", None)
    if flat is not None:
        d[flat] = 99  # imagens uniformes (cor sólida) não entram na comparação perceptual
    return d[:, 0], d.min(1), d.argmin(1)


def _thumb_img(store, rec):
    b = store.thumb(rec.id)
    return Image.open(io.BytesIO(b)).convert("RGB") if b else None


def _content_ok(store, qthumb, q, c, limit=sim.CONTENT_REJECT) -> bool:
    """Confirma pelo conteúdo da miniatura (ignora fundo/margens)."""
    if qthumb is None:
        return True
    ct = _thumb_img(store, c)
    if ct is None:
        return True
    k = sim.best_transform(q, c) if q.phash and c.phash else 0
    return sim.content_divergence(qthumb, ct, k)[0] <= limit


def find_similar(store, engine, rec, progress=None, cancelled=lambda: False,
                 workers: int | None = None) -> list[dict] | None:
    row = rec.row
    N = len(store.records)
    if not store.P_ok[row]:
        return []
    results: dict[int, dict] = {}
    q = rec
    if getattr(rec, "flat", False):
        # imagem uniforme (só branco, só preto, cor sólida): sem conteúdo para comparar;
        # apenas cópias bit a bit são relatadas
        out = []
        for j in store.sha_groups.get(rec.sha256, []):
            if j != row:
                c = store.records[j]
                out.append({"cat": sim.CAT_IDENTICAL, "score": 100.0, "detail": "Hash SHA-256 coincidente",
                            "path": c.path, "rec": c})
        return out

    # 1-2. SHA e pHash
    for j in store.sha_groups.get(rec.sha256, []):
        if j != row:
            results[j] = {"cat": sim.CAT_IDENTICAL, "score": 100.0, "detail": "Hash SHA-256 coincidente"}
    d_id, d_min, t_min = phash_distances(store, row)
    rejected: set[int] = set()  # reprovados como "mesma imagem": só podem voltar como RECORTE (geometria)
    near = np.where((d_id <= sim.PHASH_NEAR) | (d_min <= sim.PHASH_NEAR - 2))[0]
    qthumb_b = store.thumb(rec.id)
    qthumb = Image.open(io.BytesIO(qthumb_b)).convert("RGB") if qthumb_b else None
    for j in near:
        j = int(j)
        if j == row or j in results:
            continue
        c = store.records[j]
        if sim.color_distance(q.mean_rgb, c.mean_rgb) > sim.COLOR_TOL:
            continue  # estrutura parecida mas cores muito diferentes: não é a mesma imagem
        r = sim.classify(q, c, None, None, use_orb=False)
        if r is None:
            continue
        # o pHash só "vê" a estrutura geral (baixa frequência); confirma pelo CONTEÚDO na miniatura
        ct = store.thumb(c.id) if qthumb is not None else None
        if ct:
            k = sim.best_transform(q, c) if r["cat"] == sim.CAT_TRANSFORM else 0
            cthumb = Image.open(io.BytesIO(ct)).convert("RGB")
            pc, pa = sim.content_divergence(qthumb, cthumb, k)
            if pc > sim.CONTENT_REJECT:
                rejected.add(j)
                continue  # mesma "silhueta"/layout, mas conteúdo diferente
            ar_q = (q.width or 1) / (q.height or 1)
            ar_c = (c.width or 1) / (c.height or 1)
            if r["cat"] in (sim.CAT_NEAR, sim.CAT_THUMB) and \
                    (pc > sim.CONTENT_EDIT or pa > 0.4 or abs(np.log(ar_q / ar_c)) > 0.02):
                # não é cópia limpa: a geometria decide entre RECORTE leve (zoom/corte de borda)
                # e mesma imagem COM ALTERAÇÕES
                g = sim.classify_geometry(FEATS.get(q.path), FEATS.get(c.path), int(d_id[j])) \
                    if FEATS.get(q.path) and FEATS.get(c.path) else None
                if g and g["cat"] in (sim.CAT_CROP_OF, sim.CAT_CONTAINS):
                    r = g
                else:
                    r["cat"] = sim.CAT_EDITED
                    r["detail"] = f"alterações locais em ~{max(pc, pa):.1f}% ; " + r["detail"]
            else:
                r["detail"] += f"; conteúdo divergente {pc:.1f}%"
        results[j] = r
    if cancelled():
        return None

    # 3. CNN em toda a coleção
    cnn_scores = None
    cand: list[int] = []
    if engine is not None and engine.ok and store.E_ok[row] and getattr(engine, "N", 0) == N:
        cnn_scores = engine.search(store.E[row])
        cnn_scores[row] = -1
        cnn_scores[~store.E_ok] = -1
        order = np.argsort(-cnn_scores)
        top = [int(j) for j in order[:CNN_TOPK]]
        strong = [int(j) for j in order[:CNN_MAX] if cnn_scores[j] >= CNN_STRONG]
        cand = [j for j in dict.fromkeys(top + strong + sorted(rejected))
                if j not in results and (cnn_scores[j] > 0.2 or j in rejected)]
    elif engine is None or not engine.ok:
        # sem CNN: candidatos só pelos vizinhos da análise global (se houver)
        if N <= BRUTE_FORCE_MAX:  # coleção pequena: ORB contra todas
            cand = [j for j in range(N) if j != row and j not in results
                    and store.P_ok[j] and not store.flat[j]]
        else:
            cand = [store.by_id[i].row for i, _ in store.neighbors.get(rec.id, []) if i in store.by_id]
            cand = [j for j in dict.fromkeys(cand + sorted(rejected)) if j not in results]
    if cancelled():
        return None

    # 4. verificação geométrica
    qf = FEATS.get(rec.path)
    total = len(cand)

    def work(j):
        if cancelled():
            return j, None
        c = store.records[j]
        cf = FEATS.get(c.path)
        r = sim.classify(q, c, qf, cf, use_orb=True)
        if r and r["cat"] in (sim.CAT_NEAR, sim.CAT_THUMB, sim.CAT_TRANSFORM) and \
                (j in rejected or not _content_ok(store, qthumb, q, c)):
            # o pHash enganou-se quanto a "mesma imagem", mas ainda pode ser RECORTE (mesma proporção):
            # só a geometria (ORB) decide
            r = sim.classify_geometry(qf, cf) if qf is not None and cf is not None else None
        return j, r

    workers = workers or max(2, (os.cpu_count() or 4))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, j) for j in cand]
        for fut in as_completed(futs):
            j, r = fut.result()
            done += 1
            if progress and (done % 5 == 0 or done == total):
                progress(done, total)
            if (r is None and cnn_scores is not None and cnn_scores[j] >= CNN_VISUAL
                    and d_min[j] <= VISUAL_MAX_PHASH
                    and _content_ok(store, qthumb, q, store.records[j], 45.0)
                    and sim.color_distance(q.mean_rgb, store.records[j].mean_rgb) <= sim.COLOR_TOL):
                # só a CNN não basta: exige também estrutura global (pHash) e cores compatíveis
                r = {"cat": CAT_VISUAL, "score": float(cnn_scores[j] * 100),
                     "detail": f"similaridade CNN {cnn_scores[j]:.2f}; sem confirmação ORB"}
            if r:
                results[j] = r
    if cancelled():
        return None
    out = []
    for j, r in results.items():
        c = store.records[j]
        r["path"] = c.path
        r["rec"] = c
        if cnn_scores is not None:
            r["cnn"] = float(cnn_scores[j])
        out.append(r)
    out.sort(key=lambda r: (sim.CAT_ORDER.index(r["cat"]), -r["score"]))
    return out


VERIFY_GLOBAL_MIN = 0.45   # vizinhos da CNN acima disso são conferidos antes de virar selo ≈


def _verify_pairs(store, pairs, progress=None, cancelled=lambda: False):
    """Confere pares (i, j, escore) usando só as miniaturas do índice (rápido, sem reler originais)."""
    import threading
    cache_t: OrderedDict = OrderedDict()
    cache_f: OrderedDict = OrderedDict()
    lock = threading.Lock()

    def thumb(r):
        with lock:
            if r.id in cache_t:
                return cache_t[r.id]
        im = _thumb_img(store, r)
        with lock:
            cache_t[r.id] = im
            if len(cache_t) > 20000:
                cache_t.popitem(last=False)
        return im

    def feat(r):
        with lock:
            if r.id in cache_f:
                return cache_f[r.id]
        im = thumb(r)
        f = None
        if im is not None:
            xy, desc, w, h = sim.orb_features(im)
            f = sim.Feat(xy, desc, w, h)
        with lock:
            cache_f[r.id] = f
            if len(cache_f) > 20000:
                cache_f.popitem(last=False)
        return f

    seen = {}

    def check(p):
        i, j, s = p
        key = (min(i, j), max(i, j))
        if key in seen:
            return p if seen[key] else None
        a, b = store.records[i], store.records[j]
        ok = False
        if a.sha256 and a.sha256 == b.sha256:
            ok = True
        else:
            ta, tb = thumb(a), thumb(b)
            if ta is not None and tb is not None:
                k = sim.best_transform(a, b) if a.phash and b.phash else 0
                ok = sim.content_divergence(ta, tb, k)[0] <= sim.CONTENT_REJECT
                if not ok:
                    fa, fb = feat(a), feat(b)
                    ok = bool(fa and fb and sim.match_features(fa, fb))
        seen[key] = ok
        return p if ok else None

    out = []
    n = len(pairs)
    with ThreadPoolExecutor(max(2, os.cpu_count() or 4)) as ex:
        for k0 in range(0, n, 4000):
            if cancelled():
                return None
            out += [r for r in ex.map(check, pairs[k0:k0 + 4000]) if r]
            if progress:
                progress(min(n, k0 + 4000), n)
    return out


def global_analysis(store, engine, k=30, progress=None, cancelled=lambda: False):
    """Todos x todos. Com CNN: k vizinhos por imagem (GPU/multi-GPU). Sem CNN: pHash.

    Retorna dict id -> [(id_vizinho, escore 0-1)], já salvo no índice.
    """
    N = len(store.records)
    nb: dict[int, list[tuple[int, float]]] = {}
    ids = np.array([r.id for r in store.records], np.int64)
    if engine is not None and engine.ok and getattr(engine, "N", 0) == N:
        idx, sc = engine.search_all(store.E, k=k, progress=progress, cancelled=cancelled)
        if cancelled():
            return None
        thr = VERIFY_GLOBAL_MIN
        pairs = []
        for i in range(N):
            if not store.E_ok[i]:
                continue
            lst = [(int(j), float(s)) for j, s in zip(idx[i], sc[i]) if j >= 0 and store.E_ok[j]]
            nb[int(ids[i])] = [(int(ids[j]), s) for j, s in lst if s < thr]  # abaixo do limiar: só registro
            pairs += [(i, j, s) for j, s in lst if s >= thr]
        # confirma os vizinhos fortes (selos ≈): mesmo conteúdo na miniatura OU recorte (ORB nas miniaturas)
        verified = _verify_pairs(store, pairs, progress, cancelled)
        if verified is None:
            return None
        for i, j, s in verified:
            nb[int(ids[i])].append((int(ids[j]), s))
    # pHash todos x todos (identidade) — complementa a CNN e cobre o modo sem CNN
    class _Thumbs:  # cache pequeno de miniaturas decodificadas
        def __init__(self):
            self.d = OrderedDict()

        def get(self, rec):
            if rec.id in self.d:
                return self.d[rec.id]
            im = _thumb_img(store, rec)
            self.d[rec.id] = im
            if len(self.d) > 2000:
                self.d.popitem(last=False)
            return im

    thumbs = _Thumbs()
    P = store.P[:, 0]
    ok = store.P_ok & ~store.flat
    blk = 64
    for a in range(0, N, blk):
        if cancelled():
            return None
        d = np.bitwise_count(P[a:a + blk, None] ^ P[None, :]).astype(np.int16)
        d[:, ~ok] = 99
        for i in range(d.shape[0]):
            gi = a + i
            if not ok[gi]:
                continue
            js = np.where(d[i] <= sim.PHASH_NEAR)[0]
            if len(js) <= 1:
                continue
            lst = nb.setdefault(int(ids[gi]), [])
            have = {x for x, _ in lst}
            qi = store.records[gi]
            qt = None
            for j in js:
                if j != gi and int(ids[j]) not in have:
                    cj = store.records[int(j)]
                    if sim.color_distance(qi.mean_rgb, cj.mean_rgb) > sim.COLOR_TOL:
                        continue
                    if qi.sha256 != cj.sha256:  # confirma pelo conteúdo (fundo/margem não contam)
                        if qt is None:
                            qt = thumbs.get(qi)
                        ctb = thumbs.get(cj)
                        if qt is not None and ctb is not None and \
                                sim.content_divergence(qt, ctb, 0)[0] > sim.CONTENT_REJECT:
                            continue
                    lst.append((int(ids[j]), 1.0 - d[i, j] / 64))
        if progress and engine is None:
            progress(min(N, a + blk), N)
    for i, lst in nb.items():
        lst.sort(key=lambda t: -t[1])
    store.save_neighbors(nb)
    return nb
