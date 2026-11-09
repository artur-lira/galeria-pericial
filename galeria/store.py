"""Índice da pasta analisada (SQLite + matrizes em memória) e varredura em paralelo.

Escala-alvo: centenas de milhares de arquivos.
- A decodificação roda em PROCESSOS (todos os núcleos). JPEG é decodificado já reduzido.
- A CNN roda em lotes na(s) GPU(s); com várias GPUs, os lotes são distribuídos entre elas.
- A indexação é incremental: só reprocessa arquivos novos ou alterados, e pode ser interrompida
  e retomada.
- Busca textual por FTS5 (trigramas): encontra qualquer trecho de nome, caminho ou metadado.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import threading
import time
from collections import deque
from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait)
from datetime import datetime

import numpy as np
from PIL import Image

from . import engine as eng
from . import formats, loader, similarity as sim

THUMB_SIZE = 200
DECODE_SIDE = 448   # = entrada da CNN; JPEG grande é decodificado já reduzido (1/2, 1/4, 1/8)
CNN_BATCH = 32
MEAN_SAMPLES = 512
SCHEMA_VERSION = "2"
META_VERSION = "2"   # v2: datas de captura também via XMP/PNG


def cache_root() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    d = os.path.join(base, "GaleriaPericial", "indices")
    os.makedirs(d, exist_ok=True)
    return d


def db_path_for(folder: str) -> str:
    folder = os.path.normcase(os.path.abspath(folder))
    h = hashlib.sha1(folder.encode("utf-8")).hexdigest()[:12]
    name = "".join(c if c.isalnum() else "_" for c in os.path.basename(folder.rstrip("\\/")))[:40]
    d = os.path.join(cache_root(), f"{name}_{h}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "indice.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE, name TEXT, size INTEGER, mtime REAL,
    sha256 TEXT, md5 TEXT, fmt TEXT, ext_ok INTEGER,
    width INTEGER, height INTEGER, date_taken TEXT, date_file TEXT, camera TEXT,
    gps_lat REAL, gps_lon REAL, meta_json TEXT, phash BLOB, dhash TEXT,
    emb BLOB, model TEXT, emb_dist INTEGER, error TEXT, flatmask INTEGER, mean_rgb TEXT
);
CREATE INDEX IF NOT EXISTS idx_sha ON images(sha256);
CREATE TABLE IF NOT EXISTS thumbs (id INTEGER PRIMARY KEY, thumb BLOB, emb_thumb BLOB);
CREATE TABLE IF NOT EXISTS neighbors (id INTEGER PRIMARY KEY, nb BLOB, sc BLOB);
"""


# ---------------------------------------------------------------------------
# Trabalho executado nos processos de decodificação
# ---------------------------------------------------------------------------
def process_file(args):
    path, want_cnn = args
    st = os.stat(path)
    rec = {"path": path, "name": os.path.basename(path), "size": st.st_size, "mtime": st.st_mtime,
           "date_file": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
           "error": None, "phash": None, "dhash": None, "emb_dist": None}
    out = {"rec": rec, "cnn": None, "thumb": None, "emb_thumb": None, "t": {}}
    tm = out["t"]
    clk = [time.perf_counter()]

    def lap(name):  # perfil de tempo por etapa (exibido no diagnóstico)
        now = time.perf_counter()
        tm[name] = tm.get(name, 0.0) + now - clk[0]
        clk[0] = now

    try:
        with open(path, "rb") as f:
            data = f.read()
        lap("Leitura do disco")
    except OSError as e:
        rec["error"] = str(e)
        rec["meta_json"] = json.dumps({"Erro": str(e)}, ensure_ascii=False)
        return out
    rec["sha256"], rec["md5"] = loader.hashes_bytes(data)
    lap("Hash SHA-256 + MD5")
    fmt = formats.sniff_bytes(data[:64])
    rec["fmt"] = fmt or "?"
    rec["ext_ok"] = 1 if formats.describe_ext_status(path, fmt) == "coerente" else 0
    md = loader.extract_metadata(path, data, fmt)
    rec.update(width=md["width"], height=md["height"], date_taken=md["date_taken"],
               camera=md["camera"], gps_lat=md["gps"][0] if md["gps"] else None,
               gps_lon=md["gps"][1] if md["gps"] else None)
    if fmt == "RAW" or not rec["fmt"] or rec["fmt"] == "?":
        rec["fmt"] = fmt or md["format"] or "?"
    meta = md["meta"]
    meta["SHA-256"] = rec["sha256"]
    meta["MD5"] = rec["md5"]
    lap("Metadados (EXIF/XMP)")
    try:
        img = loader.to_rgb(loader.open_pil(path, DECODE_SIDE, data, fmt))
        lap("Decodificação da imagem")
        if not rec["width"]:
            rec["width"], rec["height"] = img.size
        ph = sim.dihedral_phashes(img)
        rec["phash"] = np.array(ph, dtype=np.uint64).tobytes()
        rec["dhash"] = str(sim.dhash(img))
        t = img.copy()
        t.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS, reducing_gap=2.0)
        b = io.BytesIO()
        t.save(b, "JPEG", quality=82)
        out["thumb"] = b.getvalue()
        rec["flatmask"], rec["mean_rgb"] = sim.region_flatmask(t)
        lap("pHash, miniatura, uniformidade")
        if want_cnn:
            out["cnn"] = eng.prepare_input(img)
            lap("Preparação p/ CNN")
        emb = loader.extract_embedded_thumbnail(path, data)
        if emb is not None:
            eb = io.BytesIO()
            emb.save(eb, "JPEG", quality=90)
            out["emb_thumb"] = eb.getvalue()
            eh = sim.dihedral_phashes(emb)[0]
            rec["emb_dist"] = min(sim.hamming(eh, x) for x in ph)
            meta["Miniatura EXIF embutida"] = (
                f"{emb.width}x{emb.height}; distância pHash p/ imagem principal = {rec['emb_dist']}"
                + ("  ⚠ DIVERGENTE" if rec["emb_dist"] > 14 else ""))
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        meta["Erro de decodificação"] = rec["error"]
    rec["meta_json"] = json.dumps(meta, ensure_ascii=False)
    lap("Miniatura EXIF e finalização")
    return out


# ---------------------------------------------------------------------------
# Registro leve em memória (centenas de milhares de itens)
# ---------------------------------------------------------------------------
class Rec:
    __slots__ = ("row", "id", "path", "name", "size", "sha256", "md5", "fmt", "ext_ok", "width",
                 "height", "date_taken", "date_file", "camera", "gps_lat", "gps_lon", "emb_dist",
                 "error", "flatmask", "mean_rgb", "has_emb", "_store")
    FIELDS = ("id", "path", "name", "size", "sha256", "md5", "fmt", "ext_ok", "width", "height",
              "date_taken", "date_file", "camera", "gps_lat", "gps_lon", "emb_dist", "error",
              "flatmask", "mean_rgb")

    @property
    def flat(self) -> bool:
        """Imagem inteira uniforme (cor sólida / praticamente sem conteúdo)."""
        return bool((self.flatmask or 0) & 1)

    def __getitem__(self, k):
        if k == "phash":
            return self.phash
        if k == "folder":
            return os.path.dirname(self.path)
        if k == "format":
            return self.fmt
        return getattr(self, k)

    def get(self, k, default=None):
        try:
            v = self[k]
        except AttributeError:
            return default
        return default if v is None else v

    @property
    def phash(self):
        P = self._store.P
        if P is None or not self._store.P_ok[self.row]:
            return None
        return [int(x) for x in P[self.row]]


class Store:
    def __init__(self, folder: str):
        self.folder = os.path.normpath(os.path.abspath(folder))
        self.db_path = db_path_for(self.folder)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(images)")}
        for c, t in (("flatmask", "INTEGER"), ("mean_rgb", "TEXT")):  # índices de versões anteriores
            if c not in cols:
                self.conn.execute(f"ALTER TABLE images ADD COLUMN {c} {t}")
        try:
            self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, tokenize='trigram')")
            self.has_trigram = True
        except sqlite3.OperationalError:
            self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text)")
            self.has_trigram = False
        self.conn.commit()
        self.records: list[Rec] = []
        self.by_id: dict[int, Rec] = {}
        self.P: np.ndarray | None = None
        self.P_ok: np.ndarray | None = None
        self.E: np.ndarray | None = None
        self.E_ok: np.ndarray | None = None
        self.neighbors: dict[int, list[tuple[int, float]]] = {}
        self.cnn_error = None
        self.profile = None

    # ----------------------------------------------------------------- meta
    def get_meta(self, key):
        with self.lock:
            r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def set_meta(self, key, value):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
            self.conn.commit()

    def setup_engine(self, engine: eng.Engine):
        """Recupera (ou cria) a projeção e a média usadas pelo índice."""
        if not engine.ok:
            return
        if self.get_meta("model") != eng.MODEL_ID:
            self.set_meta("model", eng.MODEL_ID)
            self.set_meta("mean", None)
            self.set_meta("proj", eng.default_projection().tobytes())
        proj = np.frombuffer(self.get_meta("proj"), np.float32).reshape(eng.DIM_IN, eng.DIM)
        mean_b = self.get_meta("mean")
        mean = np.frombuffer(mean_b, np.float32) if mean_b else None
        engine.proj = proj
        engine.mean = mean

    # ----------------------------------------------------------------- varredura
    def enumerate_files(self, recursive=True, sniff_all=True, progress=None, cancelled=lambda: False):
        known, unknown = [], []
        stack = [self.folder]
        n = 0
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if cancelled():
                            return []
                        try:
                            if e.is_dir(follow_symlinks=False):
                                if recursive:
                                    stack.append(e.path)
                                continue
                            if not e.is_file(follow_symlinks=False):
                                continue
                        except OSError:
                            continue
                        n += 1
                        if loader.is_image_file(e.name):
                            known.append(os.path.normpath(e.path))
                        elif sniff_all:
                            unknown.append(os.path.normpath(e.path))
                        if progress and n % 5000 == 0:
                            progress(0, 0, f"Listando arquivos… {n}")
            except OSError:
                continue
        if unknown:
            if progress:
                progress(0, 0, f"Verificando assinatura de {len(unknown)} arquivos sem extensão de imagem…")
            with ThreadPoolExecutor(16) as ex:
                res = list(ex.map(formats.sniff_file, unknown))
            known += [p for p, f in zip(unknown, res) if f]
        return sorted(known, key=str.lower)

    def scan(self, engine: eng.Engine | None, recursive=True, sniff_all=True, progress=None,
             cancelled=lambda: False, workers: int | None = None):
        t0 = time.time()
        files = self.enumerate_files(recursive, sniff_all, progress, cancelled)
        if cancelled():
            return
        use_cnn = bool(engine and engine.ok)
        with self.lock:
            rows = self.conn.execute("SELECT id, path, size, mtime, emb IS NOT NULL AS he, model, "
                                     "date_taken IS NULL AS nodate FROM images").fetchall()
        redo_dates = self.get_meta("meta_version") != META_VERSION
        db = {r["path"]: r for r in rows}
        present = set(files)
        removed = [r["id"] for p, r in db.items() if p not in present]
        if removed:
            with self.lock:
                for i in range(0, len(removed), 900):
                    ch = removed[i:i + 900]
                    q = ",".join("?" * len(ch))
                    for t in ("images", "thumbs", "neighbors"):
                        self.conn.execute(f"DELETE FROM {t} WHERE id IN ({q})", ch)
                    self.conn.execute(f"DELETE FROM fts WHERE rowid IN ({q})", ch)
                self.conn.commit()
        self._migrate_flatmask(progress, cancelled)
        todo = []
        for p in files:
            r = db.get(p)
            if r is None:
                todo.append(p)
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            if r["size"] != st.st_size or abs(r["mtime"] - st.st_mtime) > 1e-3:
                todo.append(p)
            elif redo_dates and r["nodate"]:  # índice antigo: tenta extrair a data por XMP/PNG
                todo.append(p)
            elif use_cnn and (not r["he"] or r["model"] != eng.MODEL_ID) and not self._row_error(p):
                todo.append(p)
        total = len(todo)
        if progress:
            progress(0, total, "")
        if total:
            self._process(todo, engine if use_cnn else None, progress, cancelled, workers)
        self.set_meta("last_scan", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if not cancelled():
            self.set_meta("meta_version", META_VERSION)
        self.scan_seconds = time.time() - t0

    def _migrate_flatmask(self, progress=None, cancelled=lambda: False):
        """Índices de versões anteriores: calcula a uniformidade a partir das miniaturas já salvas
        (rápido; não relê os arquivos originais)."""
        with self.lock:
            rows = self.conn.execute("SELECT i.id, t.thumb FROM images i JOIN thumbs t ON t.id = i.id "
                                     "WHERE i.flatmask IS NULL AND t.thumb IS NOT NULL").fetchall()
        if not rows:
            return
        n = len(rows)

        def work(r):
            try:
                return r[0], *sim.region_flatmask(Image.open(io.BytesIO(r[1])))
            except Exception:  # noqa: BLE001
                return r[0], 0, None

        done = 0
        with ThreadPoolExecutor(max(2, os.cpu_count() or 4)) as ex:
            for i in range(0, n, 2000):
                if cancelled():
                    return
                res = list(ex.map(work, rows[i:i + 2000]))
                with self.lock:
                    self.conn.executemany("UPDATE images SET flatmask=?, mean_rgb=? WHERE id=?",
                                          [(m, c, rid) for rid, m, c in res])
                    self.conn.commit()
                done += len(res)
                if progress:
                    progress(done, n, "Atualizando índice: detecção de imagens/regiões uniformes")

    def _row_error(self, path):
        with self.lock:
            r = self.conn.execute("SELECT error FROM images WHERE path=?", (path,)).fetchone()
        return bool(r and r[0])

    def _process(self, todo, engine, progress, cancelled, workers):
        workers = workers or max(1, (os.cpu_count() or 4) - 1)
        prof = self.profile = {"t0": time.perf_counter(), "t1": None, "n": 0, "total": len(todo),
                               "workers": workers, "stages": {}, "write": 0.0, "wait_gpu": 0.0,
                               "gpus": len(engine.devices) if engine else 0}
        want_cnn = engine is not None
        done = 0
        total = len(todo)
        write_buf = []
        cnn_buf = []          # (out) aguardando lote
        raw_wait = []         # (outs, raw) aguardando a média da coleção
        emb_futs = deque()
        dev_q = deque(engine.devices if engine else [])
        dev_lock = threading.Lock()

        def embed_job(outs):
            # vai para a primeira GPU livre (fila), e não em rodízio fixo
            if getattr(engine, "free_devices", None) is not None:
                raw, _dev = engine.embed_any([o["cnn"] for o in outs])
            else:
                with dev_lock:
                    dev = dev_q.popleft()
                    dev_q.append(dev)
                raw = engine.raw_descriptors([o["cnn"] for o in outs], dev)
            return outs, raw

        def flush_writes(force=False):
            if write_buf and (force or len(write_buf) >= 256):
                tw = time.perf_counter()
                self._write(write_buf)
                prof["write"] += time.perf_counter() - tw
                write_buf.clear()

        def handle_raw(outs, raw):
            if engine.mean is None:
                raw_wait.append((outs, raw))
                if sum(len(o) for o, _ in raw_wait) < MEAN_SAMPLES:
                    return
                self._init_mean(engine, raw_wait)
                pend = list(raw_wait)
                raw_wait.clear()
                for o2, r2 in pend:
                    handle_raw(o2, r2)
                return
            emb = engine.finalize(raw)
            for o, e in zip(outs, emb):
                o["rec"]["emb"] = e.tobytes()
                o["rec"]["model"] = eng.MODEL_ID
                write_buf.append(o)
            flush_writes()

        def drain_emb(block=False):
            while emb_futs and (block or emb_futs[0].done() or len(emb_futs) > 2 * len(dev_q)):
                fut = emb_futs.popleft()
                try:
                    tw = time.perf_counter()
                    outs, raw = fut.result()
                    prof["wait_gpu"] += time.perf_counter() - tw
                except Exception as e:  # noqa: BLE001 — CNN falhou: segue sem descritor
                    state["cnn_error"] = f"{type(e).__name__}: {e}"
                    state["want_cnn"] = False
                    for o in fut.outs:
                        write_buf.append(o)
                    continue
                handle_raw(outs, raw)

        state = {"want_cnn": want_cnn, "cnn_error": None}
        emb_pool = ThreadPoolExecutor(max(1, len(dev_q))) if want_cnn else None

        def submit_embed(outs):
            fut = emb_pool.submit(embed_job, outs)
            fut.outs = outs
            emb_futs.append(fut)
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                it = iter(todo)
                inflight = set()
                window = workers * 4
                exhausted = False
                while True:
                    while not exhausted and len(inflight) < window and not cancelled():
                        try:
                            p = next(it)
                        except StopIteration:
                            exhausted = True
                            break
                        f = ex.submit(process_file, (p, want_cnn))
                        f.path = p
                        inflight.add(f)
                    if not inflight:
                        break
                    fin, inflight = wait(inflight, timeout=0.5, return_when=FIRST_COMPLETED)
                    for f in fin:
                        try:
                            out = f.result()
                        except Exception as e:  # noqa: BLE001
                            out = {"rec": {"path": f.path, "name": os.path.basename(f.path), "size": -1,
                                           "mtime": 0, "error": str(e),
                                           "meta_json": json.dumps({"Erro": str(e)})},
                                   "cnn": None, "thumb": None, "emb_thumb": None}
                        for k_, v_ in (out.get("t") or {}).items():
                            prof["stages"][k_] = prof["stages"].get(k_, 0.0) + v_
                        prof["n"] += 1
                        if state["want_cnn"] and out["cnn"] is not None:
                            cnn_buf.append(out)
                            if len(cnn_buf) >= getattr(engine, "cnn_batch", CNN_BATCH):
                                submit_embed(list(cnn_buf))
                                cnn_buf.clear()
                        else:
                            write_buf.append(out)
                        done += 1
                        if progress:
                            progress(done, total, os.path.basename(f.path))
                    if want_cnn:
                        drain_emb()
                    flush_writes()
                    if cancelled() and not inflight:
                        break
            if want_cnn:
                if cnn_buf and state["want_cnn"]:
                    submit_embed(list(cnn_buf))
                elif cnn_buf:
                    write_buf.extend(cnn_buf)
                cnn_buf.clear()
                drain_emb(block=True)
                if raw_wait:  # coleção menor que MEAN_SAMPLES
                    self._init_mean(engine, raw_wait)
                    pend = list(raw_wait)
                    raw_wait.clear()
                    for o2, r2 in pend:
                        handle_raw(o2, r2)
            flush_writes(force=True)
        finally:
            if emb_pool:
                emb_pool.shutdown(wait=True)
        self.cnn_error = state["cnn_error"]
        prof["t1"] = time.perf_counter()

    def _init_mean(self, engine, raw_wait):
        allraw = np.concatenate([r for _, r in raw_wait], 0)
        mean = allraw.reshape(-1, allraw.shape[-1]).mean(0).astype(np.float32)
        engine.mean = mean
        self.set_meta("mean", mean.tobytes())

    def _write(self, outs):
        with self.lock:
            cur = self.conn.cursor()
            for o in outs:
                rec = dict(o["rec"])
                rec.setdefault("emb", None)
                rec.setdefault("model", None)
                cols = [k for k in rec.keys()]
                old = cur.execute("SELECT id FROM images WHERE path=?", (rec["path"],)).fetchone()
                if old:
                    rec["id"] = old[0]
                    cols = list(rec.keys())
                cur.execute(f"INSERT OR REPLACE INTO images ({','.join(cols)}) VALUES "
                            f"({','.join('?' * len(cols))})", [rec[c] for c in cols])
                rid = cur.lastrowid if not old else old[0]
                cur.execute("INSERT OR REPLACE INTO thumbs VALUES (?,?,?)",
                            (rid, o.get("thumb"), o.get("emb_thumb")))
                meta = json.loads(rec.get("meta_json") or "{}")
                text = rec["path"] + " " + " ".join(f"{k} {v}" for k, v in meta.items())
                cur.execute("DELETE FROM fts WHERE rowid=?", (rid,))
                cur.execute("INSERT INTO fts(rowid, text) VALUES (?,?)", (rid, text))
            self.conn.commit()

    # ----------------------------------------------------------------- carga
    def load(self):
        """Carrega registros leves + matrizes pHash e CNN para a memória."""
        with self.lock:
            cur = self.conn.execute(
                "SELECT id,path,name,size,sha256,md5,fmt,ext_ok,width,height,date_taken,date_file,"
                "camera,gps_lat,gps_lon,emb_dist,error,flatmask,mean_rgb,phash,emb FROM images "
                "ORDER BY path COLLATE NOCASE")
            rows = cur.fetchall()
        n = len(rows)
        self.records = []
        P = np.zeros((n, 8), np.uint64)
        P_ok = np.zeros(n, bool)
        E = np.zeros((n, eng.N_VEC, eng.DIM), np.int8)
        E_ok = np.zeros(n, bool)
        esz = eng.N_VEC * eng.DIM
        for i, r in enumerate(rows):
            rec = Rec()
            for k in Rec.FIELDS:
                setattr(rec, k, r[k])
            rec.row = i
            rec._store = self
            if r["phash"]:
                P[i] = np.frombuffer(r["phash"], np.uint64)
                P_ok[i] = True
            e = r["emb"]
            rec.has_emb = bool(e) and len(e) == esz
            if rec.has_emb:
                E[i] = np.frombuffer(e, np.int8).reshape(eng.N_VEC, eng.DIM)
                E_ok[i] = True
            self.records.append(rec)
        # regiões uniformes não participam da busca da CNN (vetor zerado => cosseno 0)
        fm = np.array([r.flatmask or 0 for r in self.records], np.int64)
        bits = ((fm[:, None] >> np.arange(eng.N_VEC)[None, :]) & 1).astype(bool)  # (N,15)
        E[bits] = 0
        self.flat = (fm & 1).astype(bool)
        E_ok &= ~self.flat
        self.P, self.P_ok, self.E, self.E_ok = P, P_ok, E, E_ok
        self.by_id = {r.id: r for r in self.records}
        self.sha_groups: dict[str, list[int]] = {}
        for r in self.records:
            if r.sha256:
                self.sha_groups.setdefault(r.sha256, []).append(r.row)
        self.load_neighbors()

    def load_neighbors(self):
        with self.lock:
            rows = self.conn.execute("SELECT id, nb, sc FROM neighbors").fetchall()
        self.neighbors = {}
        for r in rows:
            ids = np.frombuffer(r["nb"], np.int64)
            sc = np.frombuffer(r["sc"], np.float32)
            self.neighbors[r["id"]] = list(zip(ids.tolist(), sc.tolist()))

    def save_neighbors(self, nb: dict[int, list[tuple[int, float]]]):
        with self.lock:
            self.conn.execute("DELETE FROM neighbors")
            self.conn.executemany(
                "INSERT INTO neighbors VALUES (?,?,?)",
                [(i, np.array([a for a, _ in v], np.int64).tobytes(),
                  np.array([b for _, b in v], np.float32).tobytes()) for i, v in nb.items()])
            self.conn.commit()
        self.neighbors = nb

    # ----------------------------------------------------------------- consultas
    def text_search(self, term: str) -> set[int]:
        term = term.strip()
        with self.lock:
            if self.has_trigram and len(term) >= 3:
                q = '"' + term.replace('"', '""') + '"'
                rows = self.conn.execute("SELECT rowid FROM fts WHERE fts MATCH ?", (q,)).fetchall()
            else:
                rows = self.conn.execute("SELECT rowid FROM fts WHERE text LIKE ?",
                                         (f"%{term}%",)).fetchall()
        return {r[0] for r in rows}

    def thumb(self, rid: int) -> bytes | None:
        with self.lock:
            r = self.conn.execute("SELECT thumb FROM thumbs WHERE id=?", (rid,)).fetchone()
        return r[0] if r else None

    def emb_thumb(self, rid: int) -> bytes | None:
        with self.lock:
            r = self.conn.execute("SELECT emb_thumb FROM thumbs WHERE id=?", (rid,)).fetchone()
        return r[0] if r else None

    def meta(self, rid: int) -> dict:
        with self.lock:
            r = self.conn.execute("SELECT meta_json FROM images WHERE id=?", (rid,)).fetchone()
        return json.loads(r[0]) if r and r[0] else {}

    def clear(self):
        with self.lock:
            for t in ("images", "thumbs", "neighbors", "fts", "meta"):
                self.conn.execute(f"DELETE FROM {t}")
            self.conn.commit()
            self.conn.execute("VACUUM")

    def close(self):
        with self.lock:
            self.conn.close()
