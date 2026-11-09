"""Motor de descritores profundos (CNN) e busca vetorial, em CPU, GPU CUDA ou várias GPUs.

Descritor: ResNet-50 (ImageNet) até a camada 'layer3' (mapa 1024 canais, passo 16),
com agregação GeM por regiões (estilo R-MAC):

    índice 0  -> imagem inteira em 448 px
    índice 1  -> imagem inteira em 224 px (aproxima a escala de recortes ampliados)
    índices 2..14 -> 13 regiões (grade 2x2 com 60% do lado + grade 3x3 com 40%)

Cada vetor é centralizado (média da coleção), projetado para 256 dimensões e
quantizado em int8. A semelhança entre duas imagens é o MAIOR cosseno entre
(vetor inteiro de uma) x (qualquer vetor da outra). Assim, um recorte casa com a
região correspondente da imagem original, e vice-versa.

Esses escores servem só para encontrar CANDIDATOS rapidamente entre centenas de
milhares de imagens. A confirmação é geométrica (ORB + RANSAC), em similarity.py.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import importlib.util

# O PyTorch é importado só quando necessário: os processos de decodificação importam este
# módulo e não devem carregar o PyTorch (centenas de MB de RAM por processo).
HAS_TORCH = importlib.util.find_spec("torch") is not None
torch = None
F_ = None


def _load_torch():
    global torch, F_, HAS_TORCH
    if torch is None:
        try:
            import torch as _t
            import torch.nn.functional as _f
            torch, F_ = _t, _f
        except Exception:  # noqa: BLE001
            HAS_TORCH = False
            raise
    return torch

MODEL_ID = "r50-layer3-gem-rmac15-d256-v1"
WEIGHTS_FILE = "resnet50-0676ba61.pth"   # torchvision IMAGENET1K_V1
WEIGHTS_URL = "https://download.pytorch.org/models/" + WEIGHTS_FILE
INPUT_SIZE = 448
SMALL_SIZE = 224
STRIDE = 16
DIM_IN = 1024
DIM = 256
N_VEC = 15
FULL_IDX = [0, 1]
REGION_IDX = list(range(2, 15))
GEM_P = 3.0

_REGIONS = [(0.0, 0.0, 1.0, 1.0)]
for y in (0.0, 0.4):
    for x in (0.0, 0.4):
        _REGIONS.append((x, y, x + 0.6, y + 0.6))
for y in (0.0, 0.3, 0.6):
    for x in (0.0, 0.3, 0.6):
        _REGIONS.append((x, y, x + 0.4, y + 0.4))
assert len(_REGIONS) == 14  # + 1 escala reduzida = 15


def app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_weights() -> str | None:
    if HAS_TORCH:
        try:
            _load_torch()
        except Exception:  # noqa: BLE001
            pass
    cands = [os.path.join(app_dir(), "modelos", WEIGHTS_FILE)]
    if getattr(sys, "_MEIPASS", None):
        cands.append(os.path.join(sys._MEIPASS, "modelos", WEIGHTS_FILE))
    if HAS_TORCH:
        cands.append(os.path.join(torch.hub.get_dir(), "checkpoints", WEIGHTS_FILE))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def list_gpus() -> list[str]:
    if not HAS_TORCH:
        return []
    try:
        _load_torch()
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        return []


def gpu_info() -> list[tuple[str, float]]:
    """[(nome, memória total em GB)] de cada GPU CUDA detectada."""
    if not HAS_TORCH:
        return []
    try:
        _load_torch()
        out = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append((p.name, p.total_memory / 1024 ** 3))
        return out
    except Exception:  # noqa: BLE001
        return []


def tuning_for(mem_gb: float | None) -> dict:
    """Tamanhos de lote conforme a memória da GPU (None = CPU).

    RTX A4000 (16 GB): lote CNN 96, busca global 128 consultas x 60 mil imagens por bloco
    (~1 GB de matrizes temporárias por GPU, além do índice).
    """
    if mem_gb is None:
        return {"cnn_batch": 16, "qbatch": 128, "block": 20000, "search_block": 40000}
    if mem_gb >= 14:
        return {"cnn_batch": 96, "qbatch": 128, "block": 60000, "search_block": 100000}
    if mem_gb >= 7.5:
        return {"cnn_batch": 64, "qbatch": 64, "block": 40000, "search_block": 60000}
    return {"cnn_batch": 32, "qbatch": 32, "block": 20000, "search_block": 30000}


def default_projection(seed: int = 20261109) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((DIM_IN, DIM)))
    return q.astype(np.float32)  # (1024, 256), colunas ortonormais


def prepare_input(img):
    """PIL RGB -> (uint8 com lado maior = INPUT_SIZE, uint8 com lado maior = SMALL_SIZE).

    Feito nos PROCESSOS de decodificação, para que as threads das GPUs só transfiram e
    calculem (menos disputa pelo GIL do Python com várias GPUs).
    """
    from PIL import Image
    w, h = img.size
    s = INPUT_SIZE / max(w, h)
    big = img.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    s2 = SMALL_SIZE / max(w, h)
    small = big.resize((max(1, round(w * s2)), max(1, round(h * s2))), Image.BILINEAR)
    return np.asarray(big, dtype=np.uint8), np.asarray(small, dtype=np.uint8)


class Engine:
    """Extração de descritores e busca. devices: lista de 'cpu' ou 'cuda:N'."""

    def __init__(self, devices: list[str] | None = None, weights: str | None = None):
        self.ok = False
        self.error = ""
        self.devices: list[str] = []
        self.models = {}
        self.mean: np.ndarray | None = None
        self.proj: np.ndarray | None = None
        try:
            _load_torch()
        except Exception as e:  # noqa: BLE001
            self.error = f"PyTorch não disponível ({e})"
            return
        weights = weights or find_weights()
        if not weights:
            self.error = (f"Pesos da CNN não encontrados ({WEIGHTS_FILE}). Coloque o arquivo na "
                          f"pasta 'modelos' ao lado do programa.")
            return
        if not devices:
            n = len(list_gpus())
            devices = [f"cuda:{i}" for i in range(n)] or ["cpu"]
        self.devices = devices
        try:
            import torchvision
            sd = torch.load(weights, map_location="cpu", weights_only=True)
            for d in devices:
                m = torchvision.models.resnet50()
                m.load_state_dict(sd)
                body = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                           m.layer1, m.layer2, m.layer3).eval().to(d)
                if d.startswith("cuda"):
                    body = body.half().to(memory_format=torch.channels_last)
                self.models[d] = body
            if devices == ["cpu"]:
                torch.set_num_threads(max(1, (os.cpu_count() or 4)))
            self._locks = {d: threading.Lock() for d in self.devices}
            self._drop_failed_gpus()
            self._configure()
            self._self_test()
            self.reset_stats()
            self.ok = True
        except Exception as e:  # noqa: BLE001
            self.error = f"Falha ao iniciar a CNN em {devices}: {type(e).__name__}: {e}"
            if devices != ["cpu"]:  # GPU com problema: tenta CPU
                fb = Engine(["cpu"], weights)
                self.__dict__.update(fb.__dict__)
                if self.ok:
                    self.error = "GPU indisponível, usando CPU"
        if not hasattr(self, "_locks"):
            self._locks = {d: threading.Lock() for d in self.devices}

    def _drop_failed_gpus(self):
        """Testa a CNN em cada GPU; descarta só as que falharem (as demais continuam em uso)."""
        if self.devices == ["cpu"]:
            return
        rng = np.random.default_rng(1)
        arr = [rng.integers(0, 255, (320, 448, 3), dtype=np.uint8)]
        good, bad = [], []
        for d in self.devices:
            try:
                r = self.raw_descriptors(arr, d)
                if not np.isfinite(r).all():
                    raise ValueError("valores inválidos")
                good.append(d)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{d}: {type(e).__name__}: {e}")
                self.models.pop(d, None)
        if not good:
            raise RuntimeError("nenhuma GPU passou no autoteste — " + "; ".join(bad))
        self.devices = good
        if bad:
            self.error = "GPU(s) descartada(s): " + "; ".join(bad)

    def _configure(self):
        info = gpu_info()
        if self.devices == ["cpu"]:
            self.mem = {"cpu": None}
        else:
            self.mem = {d: info[int(d.split(":")[1])][1] for d in self.devices}
        mems = [m for m in self.mem.values() if m is not None]
        self.tuning = tuning_for(min(mems) if mems else None)
        self.cnn_batch = self.tuning["cnn_batch"]
        self.reset_stats()
        # fila de dispositivos LIVRES: cada lote vai para a primeira GPU desocupada
        self.free_devices: queue.Queue = queue.Queue()
        for d in self.devices:
            self.free_devices.put(d)

    # ----------------------------------------------------------------- estatísticas por GPU
    def reset_stats(self):
        self.stats = {d: {"imgs": 0, "batches": 0, "busy": 0.0, "queries": 0} for d in self.devices}
        self.stats_t0 = time.perf_counter()
        self._stats_lock = threading.Lock()

    def _stat(self, dev, **kw):
        with self._stats_lock:
            st = self.stats.setdefault(dev, {"imgs": 0, "batches": 0, "busy": 0.0, "queries": 0})
            for k, v in kw.items():
                st[k] += v

    def memory(self) -> dict:
        """dev -> (usada_GB_por_este_programa, ocupada_total_GB, total_GB)."""
        out = {}
        for d in self.devices:
            if not d.startswith("cuda"):
                continue
            try:
                i = int(d.split(":")[1])
                free, total = torch.cuda.mem_get_info(i)
                out[d] = (torch.cuda.memory_reserved(i) / 1024 ** 3, (total - free) / 1024 ** 3,
                          total / 1024 ** 3)
            except Exception:  # noqa: BLE001
                pass
        return out

    def embed_any(self, arrays):
        """Executa o lote na primeira GPU livre (bloqueia até haver uma)."""
        dev = self.free_devices.get()
        try:
            return self.raw_descriptors(arrays, dev), dev
        finally:
            self.free_devices.put(dev)

    def benchmark(self, seconds: float = 4.0, progress=None) -> dict:
        """Mede imagens/s de cada GPU sozinha e de todas juntas (dados sintéticos)."""
        rng = np.random.default_rng(3)
        from PIL import Image
        base = Image.fromarray(rng.integers(0, 255, (336, 448, 3), dtype=np.uint8))
        arrs = [prepare_input(base) for _ in range(self.cnn_batch)]  # igual à indexação real
        res = {"single": {}, "all": 0.0}

        def loop(dev, until, counter):
            n = 0
            while time.perf_counter() < until:
                self.raw_descriptors(arrs, dev)
                n += len(arrs)
            counter[dev] = n

        for d in self.devices:
            self.raw_descriptors(arrs, d)  # aquecimento
        steps = len(self.devices) + 1
        for k, d in enumerate(self.devices):
            if progress:
                progress(k, steps, f"Testando {d} sozinha…")
            c = {}
            t0 = time.perf_counter()
            loop(d, t0 + seconds, c)
            res["single"][d] = c[d] / (time.perf_counter() - t0)
        if progress:
            progress(steps - 1, steps, "Testando todas as GPUs juntas…")
        c = {}
        t0 = time.perf_counter()
        ths = [threading.Thread(target=loop, args=(d, t0 + seconds, c)) for d in self.devices]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        el = time.perf_counter() - t0
        res["all_per_dev"] = {d: c[d] / el for d in self.devices}
        res["all"] = sum(c.values()) / el
        return res

    def _self_test(self):
        """Executa a CNN e a busca em dados sintéticos em cada dispositivo (falha cedo se houver problema)."""
        rng = np.random.default_rng(0)
        arrs = [rng.integers(0, 255, (300, 448, 3), dtype=np.uint8),
                rng.integers(0, 255, (448, 200, 3), dtype=np.uint8)]
        for d in self.devices:
            raw = self.raw_descriptors(arrs, d)
            assert raw.shape == (2, N_VEC, DIM_IN) and np.isfinite(raw).all(), "descritor inválido"
        mean, proj = self.mean, self.proj
        self.mean, self.proj = raw.reshape(-1, DIM_IN).mean(0), default_projection()
        emb = self.finalize(raw)
        self.build_index(np.concatenate([emb] * 3))
        s = self.search(emb[0])
        assert s.shape == (6,) and s[0] > 0.97, f"busca inválida ({s[0]:.3f})"
        idx, sc = self.search_all(np.concatenate([emb] * 3), k=2)
        assert idx.shape == (6, 2)
        self.mean, self.proj = mean, proj
        self.N = 0
        self.shards = []

    def describe(self) -> str:
        if not self.ok:
            return "CNN desativada: " + self.error
        if self.devices == ["cpu"]:
            return "CNN em CPU" + (f" ({self.error})" if self.error else "")
        info = gpu_info()
        names = [f"{info[int(d.split(':')[1])][0]} ({info[int(d.split(':')[1])][1]:.0f} GB)"
                 for d in self.devices]
        txt = (f"{len(names)}× {names[0]}" if len(set(names)) == 1 else ", ".join(names))
        return f"CNN em {len(self.devices)} GPU(s): {txt}" + (f" — {self.error}" if self.error else "")

    def set_normalization(self, mean: np.ndarray, proj: np.ndarray):
        self.mean, self.proj = mean.astype(np.float32), proj.astype(np.float32)
        self._mean_t.clear()
        self._proj_t.clear()

    # ----------------------------------------------------------------- extração
    def _batch_tensor(self, arrays: list[np.ndarray], size: int, dev: str):
        n = len(arrays)
        x = np.zeros((n, size, size, 3), np.uint8)
        valid = []
        for i, a in enumerate(arrays):
            if a.shape[0] > size or a.shape[1] > size:
                import cv2
                s = size / max(a.shape[:2])
                a = cv2.resize(a, (max(1, round(a.shape[1] * s)), max(1, round(a.shape[0] * s))),
                               interpolation=cv2.INTER_AREA)
            h, w = a.shape[:2]
            x[i, :h, :w] = a
            valid.append((max(1, -(-w // STRIDE)), max(1, -(-h // STRIDE))))
        tx = torch.from_numpy(x)
        if dev.startswith("cuda"):
            tx = tx.pin_memory()
        t = tx.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        t = (t - mean) / std
        if dev.startswith("cuda"):
            t = t.half().contiguous(memory_format=torch.channels_last)
        return t, valid

    @staticmethod
    def _gem_regions(fm, valid, regions):
        """fm: (B,C,H,W) float32 >=0. Retorna (B, len(regions), C)."""
        p = GEM_P
        x = fm.clamp(min=1e-6).pow(p)
        ii = F_.pad(x.cumsum(2).cumsum(3), (1, 0, 1, 0))  # imagem integral
        B, R = fm.shape[0], len(regions)
        H, W = fm.shape[2], fm.shape[3]
        idx = np.zeros((4, B, R), np.int64)
        for b in range(B):
            fw, fh = min(valid[b][0], W), min(valid[b][1], H)
            for r, (x0, y0, x1, y1) in enumerate(regions):
                c0 = min(int(round(x0 * fw)), fw - 1)
                r0 = min(int(round(y0 * fh)), fh - 1)
                c1 = min(max(c0 + 1, int(round(x1 * fw))), fw)
                r1 = min(max(r0 + 1, int(round(y1 * fh))), fh)
                idx[:, b, r] = (r0, r1, c0, c1)
        r0, r1, c0, c1 = (torch.from_numpy(v).to(fm.device) for v in idx)
        bi = torch.arange(B, device=fm.device)[:, None].expand(B, R)
        iip = ii.permute(0, 2, 3, 1)  # (B,H+1,W+1,C)
        s = iip[bi, r1, c1] - iip[bi, r0, c1] - iip[bi, r1, c0] + iip[bi, r0, c0]
        area = ((r1 - r0) * (c1 - c0)).float()[..., None]
        out = (s / area).clamp(min=1e-12).pow(1.0 / p)
        return out

    def raw_descriptors(self, arrays: list[np.ndarray], dev: str | None = None) -> np.ndarray:
        """(B, 15, 1024) float32, normalizados L2, antes da centralização/projeção."""
        dev = dev or self.devices[0]
        model = self.models[dev]
        # cada item: (array 448, array 224) já preparados, ou só o array 448
        pairs = []
        for a in arrays:
            if isinstance(a, tuple):
                pairs.append(a)
            else:
                import cv2
                h, w = a.shape[:2]
                s = SMALL_SIZE / max(h, w)
                pairs.append((a, cv2.resize(a, (max(1, round(w * s)), max(1, round(h * s))),
                                            interpolation=cv2.INTER_AREA)))
        arrays = [p[0] for p in pairs]
        small_arrays = [p[1] for p in pairs]
        with self._locks[dev], torch.no_grad():
            t_start = time.perf_counter()
            t, valid = self._batch_tensor(arrays, INPUT_SIZE, dev)
            fm = model(t).float()
            big = self._gem_regions(fm, valid, _REGIONS)            # (B,14,C)
            t2, valid2 = self._batch_tensor(small_arrays, SMALL_SIZE, dev)
            fm2 = model(t2).float()
            small = self._gem_regions(fm2, valid2, [_REGIONS[0]])   # (B,1,C)
            d = torch.cat([big[:, :1], small, big[:, 1:]], dim=1)   # ordem: 0 448, 1 224, 2.. regiões
            d = F_.normalize(d, dim=2)
            out = d.cpu().numpy()
            if hasattr(self, "_stats_lock"):
                self._stat(dev, imgs=len(arrays), batches=1, busy=time.perf_counter() - t_start)
            return out

    def finalize(self, raw: np.ndarray) -> np.ndarray:
        """Centraliza, projeta, normaliza e quantiza: (B,15,1024) -> (B,15,256) int8."""
        v = raw - self.mean
        v = v @ self.proj
        v /= np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
        return np.clip(np.round(v * 127), -127, 127).astype(np.int8)

    def embed(self, arrays: list[np.ndarray], dev: str | None = None) -> np.ndarray:
        return self.finalize(self.raw_descriptors(arrays, dev))

    # ----------------------------------------------------------------- busca
    def build_index(self, emb: np.ndarray):
        """emb: (N,15,256) int8. Distribui as linhas entre os dispositivos."""
        self.N = emb.shape[0]
        self.shards = []
        nd = len(self.devices)
        w = np.array([self.mem.get(d) or 1.0 for d in self.devices], float)
        bounds = np.concatenate([[0], np.round(np.cumsum(w) / w.sum() * self.N)]).astype(int)
        for i, d in enumerate(self.devices):
            a, b = bounds[i], bounds[i + 1]
            part = emb[a:b]
            if d.startswith("cuda"):
                full = torch.from_numpy(np.ascontiguousarray(part[:, FULL_IDX])).to(d).half().div_(127)
                reg = torch.from_numpy(np.ascontiguousarray(part[:, REGION_IDX])).to(d).half().div_(127)
            else:  # CPU: float32 (300 mil imagens ~ 4,6 GB de RAM)
                full = torch.from_numpy(part[:, FULL_IDX].astype(np.float32) / 127)
                reg = torch.from_numpy(part[:, REGION_IDX].astype(np.float32) / 127)
            self.shards.append((d, int(a), int(b), full, reg))

    @staticmethod
    def _block_scores(q_all, q_full, full, reg, dev):
        """q_all (B,15,D) q_full (B,2,D); full (n,2,D) reg (n,13,D) -> (B,n) escore máximo."""
        dt = torch.float16 if dev.startswith("cuda") else torch.float32
        if full.dtype == torch.int8:
            full = full.to(dt) / 127
            reg = reg.to(dt) / 127
        B, n = q_all.shape[0], full.shape[0]
        s1 = (q_all.reshape(-1, DIM).to(dt) @ full.reshape(-1, DIM).T)       # (B*15, n*2)
        s1 = s1.reshape(B, N_VEC, n, 2).amax(dim=(1, 3))
        s2 = (q_full.reshape(-1, DIM).to(dt) @ reg.reshape(-1, DIM).T)        # (B*2, n*13)
        s2 = s2.reshape(B, 2, n, len(REGION_IDX)).amax(dim=(1, 3))
        return torch.maximum(s1, s2).float()

    def search(self, q_emb: np.ndarray, block: int | None = None) -> np.ndarray:
        """Escore de cada imagem do índice para UMA consulta (15,256) int8 -> (N,) float32."""
        out = np.zeros(self.N, np.float32)
        block = block or self.tuning["search_block"]

        def run(shard):
            with torch.no_grad():
                _run(shard)

        def _run(shard):
            d, a, b, full, reg = shard
            q = torch.from_numpy(q_emb.astype(np.float32) / 127).to(d)[None]
            for i in range(0, b - a, block):
                s = self._block_scores(q, q[:, FULL_IDX], full[i:i + block], reg[i:i + block], d)
                out[a + i:a + i + s.shape[1]] = s[0].cpu().numpy()

        with ThreadPoolExecutor(len(self.shards)) as ex:
            list(ex.map(run, self.shards))
        return out

    def search_all(self, emb: np.ndarray, k: int = 30, qbatch: int | None = None, block: int | None = None,
                   progress=None, cancelled=lambda: False):
        """Todos x todos: para cada imagem, os k vizinhos mais parecidos. Usa todas as GPUs.

        Retorna (idx (N,k) int32, score (N,k) float32).
        """
        gpu = self.devices[0].startswith("cuda")
        qbatch = qbatch or self.tuning["qbatch"]
        block = block or self.tuning["block"]
        N = emb.shape[0]
        k = min(k, max(1, N - 1))
        best_s = np.full((N, k), -1, np.float32)
        best_i = np.full((N, k), -1, np.int32)
        lock = threading.Lock()
        done = [0]

        def run(shard):
            with torch.no_grad():
                _run(shard)

        def _run(shard):
            d, a, b, full, reg = shard
            for q0 in range(0, N, qbatch):
                if cancelled():
                    return
                q = torch.from_numpy(emb[q0:q0 + qbatch].astype(np.float32) / 127).to(d)
                parts_s, parts_i = [], []
                for i in range(0, b - a, block):
                    s = self._block_scores(q, q[:, FULL_IDX], full[i:i + block], reg[i:i + block], d)
                    # exclui a própria imagem
                    rows = torch.arange(q0, q0 + q.shape[0], device=s.device)
                    cols = torch.arange(a + i, a + i + s.shape[1], device=s.device)
                    s[rows[:, None] == cols[None, :]] = -1
                    kk = min(k, s.shape[1])
                    v, ix = torch.topk(s, kk, dim=1)
                    parts_s.append(v)
                    parts_i.append(ix + a + i)
                v = torch.cat(parts_s, 1)
                ix = torch.cat(parts_i, 1)
                kk = min(k, v.shape[1])
                v, j = torch.topk(v, kk, dim=1)
                ix = torch.gather(ix, 1, j)
                v, ix = v.cpu().numpy(), ix.cpu().numpy().astype(np.int32)
                with lock:
                    cur_s = np.concatenate([best_s[q0:q0 + len(v)], v], 1)
                    cur_i = np.concatenate([best_i[q0:q0 + len(v)], ix], 1)
                    o = np.argsort(-cur_s, axis=1)[:, :k]
                    best_s[q0:q0 + len(v)] = np.take_along_axis(cur_s, o, 1)
                    best_i[q0:q0 + len(v)] = np.take_along_axis(cur_i, o, 1)
                    done[0] += len(v)
                    self.stats.get(d, {}).__setitem__("queries", self.stats.get(d, {}).get("queries", 0) + len(v))
                    if progress:
                        progress(done[0], N * len(self.shards))

        with ThreadPoolExecutor(len(self.shards)) as ex:
            list(ex.map(run, self.shards))
        return best_i, best_s
