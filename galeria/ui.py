"""Interface gráfica (PySide6)."""
from __future__ import annotations

import csv
import os
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
from PIL import Image
from PySide6.QtCore import (QAbstractListModel, QDate, QModelIndex, QPointF, QRect, QSize, Qt,
                            QThread, Signal, QSettings, QTimer)
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap,
                           QPolygonF, QIcon, QKeySequence)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDateEdit,
                               QDoubleSpinBox,
                               QDialog, QFileDialog, QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListView, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QSlider, QSplitter, QStyle, QStyledItemDelegate, QTableWidget,
                               QTableWidgetItem, QTabWidget, QToolBar, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from . import APP_NAME, __version__, engine as eng, loader, report, similarity as sim
from .search import CAT_VISUAL, DEFAULT_GLOBAL_THR, FEATS, find_similar, global_analysis
from .store import Store


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def pil_to_qimage(img: Image.Image) -> QImage:
    img = loader.to_rgb(img)
    data = img.tobytes("raw", "RGB")
    return QImage(data, img.width, img.height, 3 * img.width, QImage.Format_RGB888).copy()


def np_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def bytes_to_pixmap(b: bytes | None) -> QPixmap:
    pm = QPixmap()
    if b:
        pm.loadFromData(b)
    return pm


def fmt_date_br(d: str | None) -> str:
    """'2024-03-15 10:22:01' -> '15/03/2024 10:22:01'."""
    if not d:
        return "—"
    try:
        return datetime.strptime(d[:19], "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M:%S")
    except ValueError:
        return d


def human_size(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


class Task(QThread):
    """Executa uma função em segundo plano."""
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def is_cancelled(self):
        return self._cancel

    def run(self):
        try:
            self.result.emit(self.fn(*self.args, **self.kw))
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Modelo da galeria
# ---------------------------------------------------------------------------
class GalleryModel(QAbstractListModel):
    def __init__(self):
        super().__init__()
        self.db: Store | None = None
        self.items: list = []
        self.counts: dict[str, int] = {}
        self._pm_cache: OrderedDict[str, QPixmap] = OrderedDict()

    def set_items(self, items):
        self.beginResetModel()
        self.items = items
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.items)

    def pixmap(self, rec) -> QPixmap:
        pm = self._pm_cache.get(rec.id)
        if pm is None:
            pm = bytes_to_pixmap(self.db.thumb(rec.id)) if self.db else QPixmap()
            self._pm_cache[rec.id] = pm
            if len(self._pm_cache) > 4000:
                self._pm_cache.popitem(last=False)
        return pm

    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid():
            return None
        r = self.items[idx.row()]
        if role == Qt.DisplayRole:
            return r["name"]
        if role == Qt.ToolTipRole:
            dt = r.get("date_taken") or "—"
            ext = "" if r.ext_ok else "<br><b style='color:#f80'>Extensão divergente/ausente</b>"
            return (f"<b>{r['name']}</b><br>{r['folder']}<br>{r.get('width')}×{r.get('height')} · "
                    f"{r.get('format')} · {human_size(r['size'] or 0)}<br>Captura (EXIF): {dt}<br>"
                    f"Arquivo modificado: {r.get('date_file')}<br>Câmera: {r.get('camera') or '—'}" + ext)
        return None


class ThumbDelegate(QStyledItemDelegate):
    def __init__(self, model: GalleryModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.size = 160

    def sizeHint(self, option, index):
        return QSize(self.size + 16, self.size + 38)

    def paint(self, p: QPainter, option, index):
        r = self.model.items[index.row()]
        rect = option.rect.adjusted(4, 4, -4, -4)
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        selected = option.state & QStyle.State_Selected
        p.fillRect(rect, QColor("#2d5c8f") if selected else QColor("#2b2b2b"))
        pm = self.model.pixmap(r)
        area = QRect(rect.x() + 4, rect.y() + 4, rect.width() - 8, self.size)
        if not pm.isNull():
            spm = pm.scaled(area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = area.x() + (area.width() - spm.width()) // 2
            y = area.y() + (area.height() - spm.height()) // 2
            p.drawPixmap(x, y, spm)
        else:
            p.setPen(QColor("#aa6666"))
            p.drawText(area, Qt.AlignCenter, "Sem prévia\n" + (r.get("format") or ""))
        p.setPen(QColor("#e8e8e8"))
        txt_rect = QRect(rect.x() + 4, area.bottom() + 4, rect.width() - 8, 26)
        fm = p.fontMetrics()
        p.drawText(txt_rect, Qt.AlignHCenter | Qt.AlignTop,
                   fm.elidedText(r["name"], Qt.ElideMiddle, txt_rect.width()))
        # selos
        n = self.model.counts.get(r.id, 0)
        bx = rect.right() - 6
        if n:
            badge = QRect(bx - 30, rect.y() + 6, 30, 20)
            p.setBrush(QColor("#d97706"))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(badge, 8, 8)
            p.setPen(Qt.white)
            p.drawText(badge, Qt.AlignCenter, f"≈{n}")
            bx -= 34
        if r.get("emb_dist") is not None and r["emb_dist"] > 14:
            badge = QRect(bx - 22, rect.y() + 6, 22, 20)
            p.setBrush(QColor("#b91c1c"))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(badge, 8, 8)
            p.setPen(Qt.white)
            p.drawText(badge, Qt.AlignCenter, "!")
        lx = rect.x() + 6
        if r.gps_lat is not None:
            p.setPen(QColor("#22c55e"))
            p.drawText(QRect(lx, rect.y() + 6, 40, 20), Qt.AlignLeft, "GPS")
            lx += 34
        if not r.ext_ok:
            p.setPen(QColor("#fb923c"))
            p.drawText(QRect(lx, rect.y() + 6, 40, 20), Qt.AlignLeft, "EXT")
        p.restore()


# ---------------------------------------------------------------------------
# Visualizador com zoom
# ---------------------------------------------------------------------------
class ZoomView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#1b1b1b"))

    def set_pixmap(self, pm: QPixmap):
        self.item.setPixmap(pm)
        self.scene().setSceneRect(self.item.boundingRect())
        self.fit()

    def fit(self):
        self._user_zoom = False
        self.resetTransform()
        self.fitInView(self.item, Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not getattr(self, "_user_zoom", False):
            self.fitInView(self.item, Qt.KeepAspectRatio)

    def wheelEvent(self, e):
        self._user_zoom = True
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def mouseDoubleClickEvent(self, e):
        self.fit()


class ViewerDialog(QDialog):
    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(path)
        self.resize(1200, 850)
        lay = QVBoxLayout(self)
        self.view = ZoomView()
        lay.addWidget(self.view)
        bar = QHBoxLayout()
        info = QLabel("Roda do mouse: zoom · arrastar: mover · duplo clique: ajustar")
        bar.addWidget(info)
        bar.addStretch()
        b100 = QPushButton("100%")
        b100.clicked.connect(lambda: (self.view.resetTransform()))
        bfit = QPushButton("Ajustar")
        bfit.clicked.connect(self.view.fit)
        bopen = QPushButton("Abrir no Explorer")
        bopen.clicked.connect(lambda: open_in_explorer(path))
        for b in (b100, bfit, bopen):
            bar.addWidget(b)
        lay.addLayout(bar)
        try:
            self.view.set_pixmap(QPixmap.fromImage(pil_to_qimage(loader.open_pil(path))))
        except Exception as e:  # noqa: BLE001
            info.setText(f"Erro ao abrir: {e}")

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(80, self.view.fit)


def open_in_explorer(path: str):
    if os.name == "nt":
        import subprocess
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    else:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))


# ---------------------------------------------------------------------------
# Diálogo de comparação
# ---------------------------------------------------------------------------
def draw_polygon(img: Image.Image, poly_norm: np.ndarray, norm_w: int, color=(255, 210, 0)) -> QPixmap:
    """Desenha o polígono (em coords normalizadas) sobre a imagem em resolução de exibição."""
    disp = img.copy()
    disp.thumbnail((1400, 1400))
    pm = QPixmap.fromImage(pil_to_qimage(disp))
    s = disp.width / norm_w if norm_w else 1
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(*color))
    pen.setWidth(max(3, pm.width() // 250))
    p.setPen(pen)
    p.setBrush(QBrush(QColor(color[0], color[1], color[2], 50)))
    p.drawPolygon(QPolygonF([QPointF(float(x) * s, float(y) * s) for x, y in poly_norm]))
    p.end()
    return pm


SORT_KEY = 0x0102  # Qt.UserRole + 2: chave de ordenação por coluna


class SimItem(QTreeWidgetItem):
    """Item da aba Semelhantes com ordenação correta por coluna.

    - Grupos (categorias) mantêm sempre a mesma ordem, em qualquer sentido de ordenação.
    - Itens sem valor (ex.: sem data de modificação) ficam sempre no final.
    """

    def __init__(self, texts, keys=None, group_rank=None):
        super().__init__(texts)
        self.keys = keys or {}
        self.group_rank = group_rank

    def __lt__(self, other):
        tw = self.treeWidget()
        col = tw.sortColumn() if tw else 0
        desc = bool(tw) and tw.header().sortIndicatorOrder() == Qt.DescendingOrder
        if self.group_rank is not None and getattr(other, "group_rank", None) is not None:
            a, b = self.group_rank, other.group_rank
            return (a > b) if desc else (a < b)  # ordem fixa das categorias
        ka, kb = self.keys.get(col), getattr(other, "keys", {}).get(col)
        if ka is None or kb is None:
            if ka is None and kb is None:
                return self.text(0) < other.text(0)
            # vazio sempre por último (inverte conforme o sentido para continuar no fim)
            return (ka is None) if desc else (kb is None)
        return ka < kb


class CompareDialog(QDialog):
    def __init__(self, q_rec, result: dict, store: Store, parent=None):
        super().__init__(parent)
        q_path = q_rec.path
        c_path = result["path"]
        self.setWindowTitle(f"Comparação — {os.path.basename(q_path)} × {os.path.basename(c_path)}")
        self.resize(1400, 860)
        lay = QVBoxLayout(self)
        head = QLabel(f"<b>{result['cat']}</b> — confiança {result['score']:.0f}% — {result['detail']}")
        head.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(head)
        tabs = QTabWidget()
        lay.addWidget(tabs)
        try:
            qi = loader.to_rgb(loader.open_pil(q_path))
            ci = loader.to_rgb(loader.open_pil(c_path))
        except Exception as e:  # noqa: BLE001
            lay.addWidget(QLabel(f"Erro ao abrir: {e}"))
            return
        m = result.get("match")
        qf, cf = FEATS.get(q_path), FEATS.get(c_path)

        # --- lado a lado
        w = QWidget()
        h = QHBoxLayout(w)
        left, right = ZoomView(), ZoomView()
        if m and result["cat"] == sim.CAT_CROP_OF:
            lp = QPixmap.fromImage(pil_to_qimage(qi))
            rp = draw_polygon(ci, m["poly_a_in_b"], cf.w)
        elif m and result["cat"] in (sim.CAT_CONTAINS, sim.CAT_EDITED, sim.CAT_OVERLAP):
            lp = draw_polygon(qi, m["poly_b_in_a"], qf.w, (0, 200, 255))
            rp = QPixmap.fromImage(pil_to_qimage(ci))
            if result["cat"] == sim.CAT_OVERLAP:
                rp = draw_polygon(ci, m["poly_a_in_b"], cf.w)
        else:
            lp = QPixmap.fromImage(pil_to_qimage(qi))
            rp = QPixmap.fromImage(pil_to_qimage(ci))
        for title, view, pm, path, img in (("SELECIONADA", left, lp, q_path, qi),
                                           ("SEMELHANTE", right, rp, c_path, ci)):
            col = QVBoxLayout()
            lab = QLabel(f"<b>{title}</b>: {os.path.basename(path)}<br>{img.width}×{img.height} · "
                         f"{human_size(os.path.getsize(path))}<br><small>{path}</small>")
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            col.addWidget(lab)
            col.addWidget(view)
            view.set_pixmap(pm)
            h.addLayout(col)
        tabs.addTab(w, "Lado a lado (região correspondente destacada)")
        self._views = [left, right]

        # --- mapa de diferenças (alinhado)
        H = m["H"] if m else None
        if result["cat"] in (sim.CAT_TRANSFORM,):
            H = None
            from .similarity import match_features
            if qf and cf:
                mm = match_features(qf, cf)
                H = mm["H"] if mm else None
        try:
            if result["cat"] == sim.CAT_TRANSFORM and H is None:
                raise ValueError("Não foi possível alinhar a imagem transformada")
            diff, pct = sim.difference_map(qi, ci, H)
            dv = ZoomView()
            dw = QWidget()
            dl = QVBoxLayout(dw)
            dl.addWidget(QLabel(f"Áreas divergentes em <span style='color:#f55'>vermelho</span> "
                                f"(caixas amarelas). Pixels divergentes na área comum: <b>{pct:.2f}%</b>. "
                                f"Áreas escurecidas não têm correspondência. Brilho/contraste globais são "
                                f"normalizados antes da comparação."))
            dl.addWidget(dv)
            dv.set_pixmap(QPixmap.fromImage(np_to_qimage(diff)))
            tabs.addTab(dw, "Mapa de diferenças")
            self._views.append(dv)

            # --- sobreposição com transparência
            qa = sim.normalize_rgb(qi)
            ca = sim.normalize_rgb(ci)
            hq, wq = qa.shape[:2]
            import cv2
            if H is None:
                warped = cv2.resize(ca, (wq, hq))
            else:
                warped = cv2.warpPerspective(ca, np.linalg.inv(H), (wq, hq))
            bw = QWidget()
            bl = QVBoxLayout(bw)
            bv = ZoomView()
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 100)
            sl.setValue(50)

            def upd(v):
                a = v / 100
                mix = (qa.astype(np.float32) * (1 - a) + warped.astype(np.float32) * a).astype(np.uint8)
                bv.item.setPixmap(QPixmap.fromImage(np_to_qimage(mix)))

            sl.valueChanged.connect(upd)
            bl.addWidget(QLabel("Deslize para alternar entre a SELECIONADA (esquerda) e a SEMELHANTE "
                                "alinhada (direita):"))
            bl.addWidget(sl)
            bl.addWidget(bv)
            bv.set_pixmap(QPixmap.fromImage(np_to_qimage(qa)))
            upd(50)
            tabs.addTab(bw, "Sobreposição alinhada")
            self._views.append(bv)
        except Exception as e:  # noqa: BLE001
            tabs.addTab(QLabel(f"Mapa de diferenças indisponível: {e}"), "Mapa de diferenças")

        # --- metadados comparados
        mq, mc = store.meta(q_rec.id), store.meta(result["rec"].id)
        keys = list(OrderedDict.fromkeys(list(mq.keys()) + list(mc.keys())))
        t = QTableWidget(len(keys), 3)
        t.setHorizontalHeaderLabels(["Campo", "Selecionada", "Semelhante"])
        for i, k in enumerate(keys):
            a, b = mq.get(k, ""), mc.get(k, "")
            for j, v in enumerate((k, a, b)):
                it = QTableWidgetItem(str(v))
                if j and a != b:
                    it.setBackground(QColor("#5a3a00"))
                t.setItem(i, j, it)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        t.setColumnWidth(0, 260)
        t.setColumnWidth(1, 520)
        t.setColumnWidth(2, 520)
        tabs.addTab(t, "Metadados comparados (divergências destacadas)")

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(80, lambda: [v.fit() for v in self._views])


# ---------------------------------------------------------------------------
# Diagnóstico das GPUs
# ---------------------------------------------------------------------------
class GpuDialog(QDialog):
    """Mostra, em tempo real, quanto cada GPU trabalhou e permite um teste de carga."""

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.setWindowTitle("Diagnóstico das GPUs")
        self.resize(1080, 640)
        lay = QVBoxLayout(self)
        self.lbl = QLabel()
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        self.tbl = QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels(["Dispositivo", "Placa", "VRAM do programa", "VRAM total ocupada",
                                            "Imagens (CNN)", "Lotes", "Tempo ativo", "Consultas (global)"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for c, wd in enumerate((90, 200, 130, 150, 120, 70, 90)):
            self.tbl.setColumnWidth(c, wd)
        lay.addWidget(self.tbl)
        self.prof = QLabel()
        self.prof.setWordWrap(True)
        self.prof.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.prof)
        self.res = QLabel()
        self.res.setWordWrap(True)
        self.res.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.res)
        hb = QHBoxLayout()
        self.b_bench = QPushButton("Teste de carga (≈20 s)")
        self.b_bench.setToolTip("Mede imagens/s de cada GPU sozinha e de todas juntas")
        self.b_bench.clicked.connect(self.bench)
        b_zero = QPushButton("Zerar contadores")
        b_zero.clicked.connect(lambda: self.win.engine and self.win.engine.reset_stats())
        hb.addWidget(self.b_bench)
        hb.addWidget(b_zero)
        hb.addStretch()
        hb.addWidget(QLabel("<small>Conferência externa: <code>nvidia-smi -l 1</code> no Prompt. "
                            "O Gerenciador de Tarefas do Windows não mostra CUDA no gráfico “3D”: "
                            "troque um gráfico da GPU para “Cuda” ou “Compute_0”.</small>"))
        lay.addLayout(hb)
        self.timer = QTimer(self, interval=1000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def refresh(self):
        e = self.win.engine
        if not e or not e.ok:
            self.lbl.setText("CNN desativada: " + (e.error if e else "motor não carregado"))
            return
        info = eng.gpu_info()
        self.lbl.setText(f"<b>{e.describe()}</b><br>GPUs detectadas no sistema: {len(info)} · "
                         f"em uso pelo programa: {', '.join(e.devices)} · lote CNN: {e.cnn_batch} imagens")
        mem = e.memory()
        devs = list(e.devices)
        self.tbl.setRowCount(len(devs))
        tot = sum(e.stats.get(d, {}).get("imgs", 0) for d in devs) or 1
        for i, d in enumerate(devs):
            st = e.stats.get(d, {})
            name = info[int(d.split(":")[1])][0] if d.startswith("cuda") and info else "CPU"
            m = mem.get(d)
            vals = [d, name, f"{m[0]:.2f} GB" if m else "—", f"{m[1]:.1f} / {m[2]:.1f} GB" if m else "—",
                    f"{st.get('imgs', 0):,} ({st.get('imgs', 0) / tot * 100:.0f}%)".replace(",", "."),
                    f"{st.get('batches', 0):,}".replace(",", "."), f"{st.get('busy', 0):.1f} s",
                    f"{st.get('queries', 0):,}".replace(",", ".")]
            for j, v in enumerate(vals):
                self.tbl.setItem(i, j, QTableWidgetItem(str(v)))
        self.tbl.setMaximumHeight(36 + 30 * max(1, len(devs)))
        self.prof.setText(self._profile_html(e))

    def _profile_html(self, e) -> str:
        """Perfil da indexação: onde o tempo está sendo gasto (CPU, disco, gravação ou GPU)."""
        st = self.win.store
        p = getattr(st, "profile", None) if st else None
        if not p or not p["n"]:
            return "<i>Perfil da indexação: disponível durante/depois de uma indexação com arquivos novos.</i>"
        wall = (p["t1"] or time.perf_counter()) - p["t0"]
        n = p["n"]
        rate = n / wall if wall > 0 else 0
        cpu_ms = sum(p["stages"].values()) / n * 1000
        cap = p["workers"] / (cpu_ms / 1000) if cpu_ms else 0
        busy = sum(e.stats.get(d, {}).get("busy", 0) for d in e.devices)
        gpu_occ = busy / (wall * max(1, len(e.devices))) * 100 if e.devices[0].startswith("cuda") else 0
        rows = "".join(
            f"<tr><td>{k}</td><td align='right'>{v / n * 1000:.1f} ms</td>"
            f"<td align='right'>{v / max(1e-9, sum(p['stages'].values())) * 100:.0f}%</td></tr>"
            for k, v in sorted(p["stages"].items(), key=lambda kv: -kv[1]))
        top = max(p["stages"].items(), key=lambda kv: kv[1])[0] if p["stages"] else "—"
        wr = p["write"] / wall * 100
        wg = p["wait_gpu"] / wall * 100
        if wr > 50:
            gargalo = "gravação no índice (SQLite/FTS) — disco do perfil do usuário"
        elif wg > 30:
            gargalo = "GPU"
        elif top == "Leitura do disco":
            gargalo = "leitura do DISCO onde estão as imagens"
        else:
            gargalo = f"CPU — etapa mais cara: {top}"
        estado = "em andamento" if p["t1"] is None else "concluída"
        return (f"<b>Perfil da indexação ({estado})</b>: {n:,} de {p['total']:,} arquivos em {wall:.0f} s → "
                f"<b>{rate:.0f} img/s</b> · {p['workers']} processos de decodificação · "
                f"capacidade estimada da CPU: {cap:.0f} img/s · ocupação média das GPUs: <b>{gpu_occ:.0f}%</b> · "
                f"gravação no índice: {wr:.0f}% do tempo · espera por GPU: {wg:.0f}% do tempo<br>"
                f"<b>Gargalo provável: {gargalo}</b>"
                f"<table cellspacing='0' cellpadding='2' style='margin-top:4px'>"
                f"<tr><th align='left'>Etapa (por imagem, em cada processo)</th><th>tempo médio</th>"
                f"<th>% da CPU</th></tr>{rows}</table>").replace(",", ".")

    def bench(self):
        e = self.win.engine
        if not e or not e.ok:
            return
        if any(t and t.isRunning() for t in (self.win.scan_task, self.win.global_task)):
            QMessageBox.information(self, APP_NAME, "Aguarde a indexação/análise terminar para medir.")
            return
        self.b_bench.setEnabled(False)
        self.res.setText("Executando teste de carga…")
        t = Task(lambda: None)
        t.fn = lambda: e.benchmark(4.0, progress=lambda a, b, s: t.progress.emit(a, b, s))
        t.progress.connect(lambda a, b, s: self.res.setText(s))
        t.result.connect(self._bench_done)
        t.error.connect(lambda m: (self.res.setText(f"Erro: {m}"), self.b_bench.setEnabled(True)))
        self._t = t
        t.start()

    def _bench_done(self, r):
        self.b_bench.setEnabled(True)
        single = r["single"]
        best = max(single.values()) if single else 1
        lines = [f"{d} sozinha: <b>{v:.0f} img/s</b>" for d, v in single.items()]
        per = ", ".join(f"{d} {v:.0f}" for d, v in r["all_per_dev"].items())
        lines.append(f"<b>Todas juntas: {r['all']:.0f} img/s</b> ({per}) — ganho de "
                     f"<b>{r['all'] / best:.2f}×</b> sobre a melhor GPU sozinha (ideal: {len(single)}×)")
        self.res.setText("<br>".join(lines) + "<br><small>A indexação real costuma ser limitada pela CPU/disco "
                         "(leitura, hash e decodificação), e não pela GPU.</small>")


# ---------------------------------------------------------------------------
# Janela principal
# ---------------------------------------------------------------------------
SPECIAL_FILTERS = ["Todas as imagens", "Com semelhantes (≈)", "Com GPS", "Sem data EXIF",
                   "Miniatura EXIF divergente (!)", "Duplicatas exatas (SHA-256)",
                   "Extensão divergente ou ausente", "Erros de leitura", "Imagem uniforme (cor sólida/vazia)"]
DATE_FIELDS = ["captura (EXIF/XMP) ou, se ausente, arquivo", "somente captura (EXIF/XMP)",
               "somente modificação do arquivo"]
SORTS = ["Caminho", "Nome", "Data de captura", "Data do arquivo", "Tamanho", "Resolução", "Formato"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1600, 950)
        self.settings = QSettings("PoliciaCientifica", "GaleriaPericial")
        self.store: Store | None = None
        self.engine: eng.Engine | None = None
        self.engine_ready = False
        self.folder: str | None = None
        self.current = None
        self.tasks: list[Task] = []
        self.scan_task = self.sim_task = self.global_task = self.report_task = None
        self.sim_results: list[dict] = []
        self._sim_cache: dict[int, list] = {}
        self._build_ui()
        self.start_engine()
        last = self.settings.value("last_folder")
        if last and os.path.isdir(last):
            # ao iniciar, reabre a última pasta exibindo todas as imagens, sem filtro
            self.pending_folder = last
            self.lbl_folder.setText(f"Abrindo a última pasta: {last}…")

    # ---------------- utilidades
    def run_task(self, fn, *a, on_result=None, on_progress=None, on_error=None, **kw) -> Task:
        t = Task(fn, *a, **kw)
        if on_result:
            t.result.connect(on_result)
        if on_progress:
            t.progress.connect(on_progress)
        t.error.connect(on_error or (lambda e: QMessageBox.critical(self, APP_NAME, e)))
        self.tasks = [x for x in self.tasks if x.isRunning()] + [t]
        t.start()
        return t

    def busy(self, on: bool, text=""):
        self.progress.setVisible(on)
        self.btn_cancel.setVisible(on)
        if on:
            self.progress.setRange(0, 0)
            self.progress.setFormat(text)

    def _prog(self, d, n, text=""):
        if n <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, n)
            self.progress.setValue(d)
        self.progress.setFormat((text[:60] + " — " if text else "") + ("%v/%m" if n > 0 else ""))
        if text and n <= 0:
            self.statusBar().showMessage(text)

    # ---------------- construção
    def _build_ui(self):
        tb = QToolBar("Principal")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        st = self.style()

        def act(icon, text, slot, shortcut=None):
            a = QAction(st.standardIcon(icon) if icon is not None else QIcon(), text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(shortcut)
            tb.addAction(a)
            return a

        act(QStyle.SP_DirOpenIcon, "Abrir pasta…", self.choose_folder, QKeySequence.Open)
        act(QStyle.SP_BrowserReload, "Reindexar", lambda: self.folder and self.load_folder(self.folder), "F5")
        self.chk_rec = QCheckBox("Subpastas")
        self.chk_rec.setChecked(True)
        self.chk_rec.setToolTip("A análise de similaridade sempre considera todos os arquivos indexados.")
        tb.addWidget(self.chk_rec)
        self.chk_sniff = QCheckBox("Detectar imagens sem extensão")
        self.chk_sniff.setChecked(True)
        self.chk_sniff.setToolTip("Lê a assinatura (magic bytes) de todos os arquivos cuja extensão não é de "
                                  "imagem, inclusive arquivos sem extensão ou renomeados.")
        tb.addWidget(self.chk_sniff)
        tb.addSeparator()
        act(QStyle.SP_FileDialogContentsView, "Análise global", self.run_global)
        act(QStyle.SP_FileDialogDetailedView, "Relatório das selecionadas", self.make_report, "Ctrl+R")
        act(QStyle.SP_DriveHDIcon, "Exportar c/ semelhantes", self.export_with_similar, "Ctrl+E")
        act(QStyle.SP_DialogSaveButton, "CSV lista", self.export_csv)
        act(None, "CSV semelhantes", self.export_similar_csv)
        tb.addSeparator()
        tb.addWidget(QLabel(" Processamento: "))
        self.cb_device = QComboBox()
        gpus = eng.gpu_info()
        self.cb_device.addItem(f"Automático — todas as GPUs detectadas ({len(gpus)})" if gpus
                               else "Automático (sem GPU → CPU)", "auto")
        self.cb_device.addItem("CPU", "cpu")
        if len(gpus) <= 4:  # todas as combinações (ex.: 3 GPUs -> 7 opções)
            from itertools import combinations
            for k in range(len(gpus), 0, -1):
                for comb in combinations(range(len(gpus)), k):
                    label = " + ".join(f"GPU {i}" for i in comb)
                    if k == 1:
                        n, m = gpus[comb[0]]
                        label += f": {n} ({m:.0f} GB)"
                    self.cb_device.addItem(label, ",".join(f"cuda:{i}" for i in comb))
        else:
            self.cb_device.addItem(f"Todas as GPUs ({len(gpus)})", "auto")
            for i, (n, m) in enumerate(gpus):
                self.cb_device.addItem(f"GPU {i}: {n} ({m:.0f} GB)", f"cuda:{i}")
        saved = self.settings.value("device", "auto")
        i = self.cb_device.findData(saved)
        self.cb_device.setCurrentIndex(max(0, i))
        self.cb_device.currentIndexChanged.connect(self._device_changed)
        self.cb_device.setMaximumWidth(300)
        tb.addWidget(self.cb_device)
        tb.addWidget(QLabel(" Limiar ≈: "))
        self.sp_thr = QDoubleSpinBox()
        self.sp_thr.setRange(0.30, 0.99)
        self.sp_thr.setSingleStep(0.02)
        self.sp_thr.setValue(float(self.settings.value("thr", DEFAULT_GLOBAL_THR)))
        self.sp_thr.setToolTip("Similaridade mínima (CNN) para contar um vizinho no selo ≈ da análise global")
        self.sp_thr.valueChanged.connect(self._update_counts)
        tb.addWidget(self.sp_thr)
        tb.addSeparator()
        act(QStyle.SP_ComputerIcon, "GPUs", self.show_gpu_dialog)
        act(QStyle.SP_TrashIcon, "Limpar índice", self.clear_cache)
        act(QStyle.SP_MessageBoxInformation, "Sobre", self.about)

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout()
        self.lbl_folder = QLabel("Nenhuma pasta aberta. Use “Abrir pasta…” (Ctrl+O).")
        top.addWidget(self.lbl_folder, 1)
        top.addWidget(QLabel("Miniaturas:"))
        self.sl_size = QSlider(Qt.Horizontal)
        self.sl_size.setRange(80, 320)
        self.sl_size.setValue(160)
        self.sl_size.setFixedWidth(140)
        self.sl_size.valueChanged.connect(self._resize_thumbs)
        top.addWidget(self.sl_size)
        v.addLayout(top)

        f = QHBoxLayout()
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText(
            "Buscar em nome, caminho e metadados — vários termos = E; “-termo” exclui. "
            "Ex.: canon 2023 -whatsapp | photoshop | GPS")
        self.ed_search.setClearButtonEnabled(True)
        self._search_timer = QTimer(self, singleShot=True, interval=350)
        self._search_timer.timeout.connect(self.apply_filters)
        self.ed_search.textChanged.connect(lambda: self._search_timer.start())
        f.addWidget(self.ed_search, 3)
        self.cb_format = QComboBox()
        self.cb_format.addItem("Todos os formatos")
        self.cb_format.currentIndexChanged.connect(self.apply_filters)
        f.addWidget(self.cb_format)
        self.cb_special = QComboBox()
        self.cb_special.addItems(SPECIAL_FILTERS)
        self.cb_special.currentIndexChanged.connect(self.apply_filters)
        f.addWidget(self.cb_special)
        self.chk_date = QCheckBox("Período:")
        self.chk_date.setToolTip("Alterar qualquer data ativa o filtro automaticamente.")
        self.chk_date.toggled.connect(self._date_toggled)
        f.addWidget(self.chk_date)
        self.cb_datefield = QComboBox()
        self.cb_datefield.addItems(DATE_FIELDS)
        self.cb_datefield.setToolTip("Qual data considerar. A data de captura vem do EXIF "
                                     "(DateTimeOriginal/Digitized/DateTime) ou, na falta, do XMP/PNG.")
        self.cb_datefield.currentIndexChanged.connect(self._date_field_changed)
        f.addWidget(self.cb_datefield)
        self.de_from = QDateEdit()
        self.de_to = QDateEdit()
        for d, lab in ((self.de_from, "de"), (self.de_to, "até")):
            f.addWidget(QLabel(lab))
            d.setCalendarPopup(True)
            d.setDisplayFormat("dd/MM/yyyy")
            d.setDateRange(QDate(1900, 1, 1), QDate(2100, 12, 31))
            d.dateChanged.connect(self._date_edited)
            f.addWidget(d)
        self._set_date_range_to_collection()
        b_clear = QPushButton("Limpar filtros")
        b_clear.setToolTip("Volta a exibir todas as imagens")
        b_clear.clicked.connect(self.reset_filters)
        f.addWidget(b_clear)
        f.addWidget(QLabel("Ordenar:"))
        self.cb_sort = QComboBox()
        self.cb_sort.addItems(SORTS)
        self.cb_sort.currentIndexChanged.connect(self.apply_filters)
        f.addWidget(self.cb_sort)
        self.chk_desc = QCheckBox("Decrescente")
        self.chk_desc.toggled.connect(self.apply_filters)
        f.addWidget(self.chk_desc)
        v.addLayout(f)

        split = QSplitter(Qt.Horizontal)
        v.addWidget(split, 1)
        self.model = GalleryModel()
        self.list = QListView()
        self.list.setViewMode(QListView.IconMode)
        self.list.setResizeMode(QListView.Adjust)
        self.list.setMovement(QListView.Static)
        self.list.setUniformItemSizes(True)
        self.list.setLayoutMode(QListView.Batched)
        self.list.setBatchSize(2000)
        self.list.setSpacing(2)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setModel(self.model)
        self.delegate = ThumbDelegate(self.model, self.list)
        self.list.setItemDelegate(self.delegate)
        self.list.setStyleSheet("QListView{background:#1e1e1e;}")
        self.list.selectionModel().currentChanged.connect(self.on_current)
        self.list.selectionModel().selectionChanged.connect(self._sel_changed)
        self.list.doubleClicked.connect(lambda i: ViewerDialog(self.model.items[i.row()].path, self).exec())
        split.addWidget(self.list)

        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        self.lbl_preview = QLabel("Selecione uma imagem")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumHeight(260)
        self.lbl_preview.setStyleSheet("background:#151515;color:#888;")
        self.lbl_preview.mouseDoubleClickEvent = lambda e: (
            ViewerDialog(self.current.path, self).exec() if self.current else None)
        sv.addWidget(self.lbl_preview, 2)
        self.lbl_info = QLabel()
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sv.addWidget(self.lbl_info)
        self.tabs = QTabWidget()
        sv.addWidget(self.tabs, 3)

        simw = QWidget()
        sl = QVBoxLayout(simw)
        hs = QHBoxLayout()
        self.lbl_sim = QLabel("Comparação com TODOS os arquivos da pasta e subpastas.")
        hs.addWidget(self.lbl_sim, 1)
        self.chk_group = QCheckBox("Agrupar por tipo de relação")
        self.chk_group.setToolTip("Desmarcado: lista única em ordem cronológica (linha do tempo das cópias).")
        self.chk_group.setChecked(self.settings.value("sim_group_v2", "false") in (True, "true"))
        self.chk_group.toggled.connect(self._group_toggled)
        hs.addWidget(self.chk_group)
        sl.addLayout(hs)
        self.sim_progress = QProgressBar()
        self.sim_progress.setVisible(False)
        sl.addWidget(self.sim_progress)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Imagem", "Confiança", "Data de modificação", "Tamanho", "Detalhe"])
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(2, Qt.AscendingOrder)  # padrão: data de modificação crescente
        self.tree.header().setToolTip("Clique no título de uma coluna para reordenar. "
                                      "Padrão: data de modificação, da mais antiga para a mais recente; "
                                      "sem data ficam no final.")
        self.tree.setIconSize(QSize(80, 80))
        self.tree.setColumnWidth(0, 250)
        self.tree.setColumnWidth(1, 75)
        self.tree.setColumnWidth(2, 135)
        self.tree.setColumnWidth(3, 80)
        self.tree.itemDoubleClicked.connect(self.on_similar_open)
        sl.addWidget(self.tree)
        b_exp = QPushButton("Exportar esta imagem e as semelhantes para uma pasta…")
        b_exp.setToolTip("Copia a imagem selecionada e todas as semelhantes listadas (com conferência de "
                         "SHA-256 e manifesto.csv). Com várias imagens selecionadas na galeria, exporta cada "
                         "uma com as suas semelhantes. Atalho: Ctrl+E")
        b_exp.clicked.connect(self.export_with_similar)
        sl.addWidget(b_exp)
        sl.addWidget(QLabel("<small>Duplo clique abre a comparação (região do recorte, mapa de diferenças, "
                            "sobreposição e metadados confrontados).</small>"))
        self.tabs.addTab(simw, "Semelhantes")

        mw = QWidget()
        ml = QVBoxLayout(mw)
        self.ed_meta = QLineEdit()
        self.ed_meta.setPlaceholderText("Filtrar campos…")
        self.ed_meta.textChanged.connect(self._filter_meta)
        ml.addWidget(self.ed_meta)
        self.tbl_meta = QTableWidget(0, 2)
        self.tbl_meta.setHorizontalHeaderLabels(["Campo", "Valor"])
        self.tbl_meta.horizontalHeader().setStretchLastSection(True)
        self.tbl_meta.setColumnWidth(0, 230)
        self.tbl_meta.verticalHeader().setVisible(False)
        self.tbl_meta.setEditTriggers(QAbstractItemView.NoEditTriggers)
        ml.addWidget(self.tbl_meta)
        self.tabs.addTab(mw, "Metadados")

        ew = QWidget()
        el = QVBoxLayout(ew)
        self.lbl_emb_info = QLabel()
        self.lbl_emb_info.setWordWrap(True)
        el.addWidget(self.lbl_emb_info)
        eh = QHBoxLayout()
        self.lbl_emb = QLabel()
        self.lbl_emb.setAlignment(Qt.AlignCenter)
        self.lbl_main_small = QLabel()
        self.lbl_main_small.setAlignment(Qt.AlignCenter)
        eh.addWidget(self.lbl_emb)
        eh.addWidget(self.lbl_main_small)
        el.addLayout(eh, 1)
        self.tabs.addTab(ew, "Miniatura EXIF embutida")
        split.addWidget(side)
        split.setSizes([1000, 600])

        self.progress = QProgressBar()
        self.progress.setMinimumWidth(420)
        self.progress.setVisible(False)
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self.cancel_all)
        self.lbl_engine = QLabel("Carregando CNN…")
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.btn_cancel)
        self.statusBar().addPermanentWidget(self.lbl_engine)
        self._gpu_timer = QTimer(self, interval=1500)
        self._gpu_timer.timeout.connect(self._gpu_status)
        self._gpu_timer.start()

    # ---------------- motor (CNN / GPU)
    def _devices(self):
        mode = self.cb_device.currentData()
        n = len(eng.gpu_info())
        if mode in ("auto", "all"):
            return [f"cuda:{i}" for i in range(n)] or ["cpu"]
        devs = [d for d in mode.split(",") if d == "cpu" or int(d.split(":")[1]) < n]
        return devs or ["cpu"]

    def start_engine(self):
        self.engine_ready = False
        devs = self._devices()
        self.lbl_engine.setText("Carregando CNN…")

        def mk():
            return eng.Engine(devs)

        def done(e):
            self.engine = e
            self.engine_ready = True
            self.lbl_engine.setText(e.describe() + f" · HEIC {'✓' if loader.HAS_HEIF else '✗'}"
                                    f" · RAW {'✓' if loader.HAS_RAW else '✗'}")
            if not e.ok:
                self.lbl_engine.setStyleSheet("color:#f80")
            if self.store and self.store.records:
                self._build_vector_index()
            if getattr(self, "pending_folder", None) and not (self.scan_task and self.scan_task.isRunning()):
                f, self.pending_folder = self.pending_folder, None
                self.load_folder(f)

        self.run_task(mk, on_result=done)

    def show_gpu_dialog(self):
        if getattr(self, "_gpu_dlg", None) is None:
            self._gpu_dlg = GpuDialog(self, self)
        self._gpu_dlg.show()
        self._gpu_dlg.raise_()

    def _gpu_status(self):
        """Durante indexação/análise: quantas imagens cada GPU já processou (barra de status)."""
        e = self.engine
        busy = any(t and t.isRunning() for t in (self.scan_task, self.global_task))
        if not (e and e.ok and busy and len(e.devices) > 1):
            return
        parts = []
        for d in e.devices:
            st = e.stats.get(d, {})
            n = st.get("imgs", 0) or st.get("queries", 0)
            parts.append(f"{d.replace('cuda:', 'GPU ')}: {n:,}".replace(",", "."))
        self.lbl_engine.setText(" · ".join(parts))

    def _device_changed(self):
        self.settings.setValue("device", self.cb_device.currentData())
        self.start_engine()

    def _build_vector_index(self):
        if not (self.engine and self.engine.ok and self.store):
            return
        if self.store.get_meta("model") == eng.MODEL_ID:
            self.store.setup_engine(self.engine)
        self.statusBar().showMessage("Carregando vetores na memória/GPU…")
        self.run_task(self.engine.build_index, self.store.E,
                      on_result=lambda _: self.statusBar().showMessage(
                          f"Índice vetorial pronto: {len(self.store.records)} imagens em "
                          f"{', '.join(self.engine.devices)}"))

    # ---------------- pasta / varredura
    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Escolha a pasta de imagens", self.folder or "")
        if d:
            self.load_folder(d)

    def cancel_all(self):
        for t in (self.scan_task, self.global_task, self.report_task):
            if t and t.isRunning():
                t.cancel()

    def load_folder(self, folder: str):
        if self.scan_task and self.scan_task.isRunning():
            return
        if not self.engine_ready:
            QMessageBox.information(self, APP_NAME, "Aguarde o carregamento do motor de análise (CNN).")
            return
        folder = os.path.normpath(folder)
        if self.store is None or self.store.folder != os.path.abspath(folder):
            if self.store:
                self.store.close()
            self.store = Store(folder)
            self.model.db = self.store
            self.model._pm_cache.clear()
        if self.folder and os.path.normcase(os.path.abspath(self.folder)) != os.path.normcase(os.path.abspath(folder)):
            self.current = None
        self.folder = folder
        self.settings.setValue("last_folder", folder)
        self.store.setup_engine(self.engine)
        self.lbl_folder.setText(f"Pasta: <b>{folder}</b> — indexando…")
        if self.engine and self.engine.ok:
            self.engine.reset_stats()
        self._sim_cache.clear()
        self.busy(True, "Listando arquivos…")
        t = Task(lambda: None)
        t.fn = lambda: (self.store.scan(self.engine, self.chk_rec.isChecked(), self.chk_sniff.isChecked(),
                                        progress=lambda d, n, name: t.progress.emit(d, n, name),
                                        cancelled=t.is_cancelled), self.store.load())
        t.progress.connect(self._prog)
        t.result.connect(self._on_scan_done)
        t.error.connect(lambda e: (self.busy(False), QMessageBox.critical(self, APP_NAME, f"Erro na indexação:\n{e}")))
        self.scan_task = t
        self.tasks.append(t)
        t.start()

    def _restore_engine_label(self):
        if self.engine:
            self.lbl_engine.setText(self.engine.describe() + f" · HEIC {'✓' if loader.HAS_HEIF else '✗'}"
                                    f" · RAW {'✓' if loader.HAS_RAW else '✗'}")

    def _on_scan_done(self, _):
        self.busy(False)
        self._restore_engine_label()
        recs = self.store.records
        fmts = sorted({(r.fmt or "?").upper() for r in recs})
        self.cb_format.blockSignals(True)
        self.cb_format.clear()
        self.cb_format.addItem("Todos os formatos")
        self.cb_format.addItems(fmts)
        self.cb_format.blockSignals(False)
        n_noemb = sum(1 for r in recs if not r.has_emb and not r.error)
        n_ext = sum(1 for r in recs if not r.ext_ok)
        secs = getattr(self.store, "scan_seconds", 0)
        self.lbl_folder.setText(
            f"Pasta: <b>{self.folder}</b> — {len(recs)} imagens ({n_ext} sem extensão/divergentes) · "
            f"indexação {secs:.0f}s" + (f" · <span style='color:#f80'>{n_noemb} sem descritor CNN</span>"
                                         if n_noemb and self.engine.ok else ""))
        self.model._pm_cache.clear()
        self.current = None
        self._compute_counts()
        self.reset_filters()  # padrão: todas as imagens, sem filtro
        self._build_vector_index()
        if getattr(self.store, "cnn_error", None):
            QMessageBox.warning(self, APP_NAME, "A CNN falhou durante a indexação; as imagens restantes "
                                f"ficaram só com pHash/ORB.\n\n{self.store.cnn_error}")
        if self.scan_task and self.scan_task.is_cancelled():
            self.statusBar().showMessage("Indexação interrompida — o que já foi processado está salvo. "
                                         "F5 continua de onde parou.")

    # ---------------- análise global
    def run_global(self):
        if not self.store or not self.store.records:
            return
        if self.engine and self.engine.ok and getattr(self.engine, "N", 0) != len(self.store.records):
            QMessageBox.information(self, APP_NAME, "Aguarde o índice vetorial terminar de carregar.")
            return
        n = len(self.store.records)
        dev = ", ".join(self.engine.devices) if self.engine and self.engine.ok else "CPU (só pHash)"
        if QMessageBox.question(
                self, APP_NAME, f"Comparar todas as {n} imagens entre si usando {dev}?\n\n"
                "Isso alimenta os selos ≈ e o filtro “Com semelhantes”. Em CPU e coleções grandes pode levar "
                "horas; com GPU, minutos.") != QMessageBox.Yes:
            return
        self.busy(True, "Análise global…")
        if self.engine and self.engine.ok:
            self.engine.reset_stats()
        t = Task(lambda: None)
        t.fn = lambda: global_analysis(self.store, self.engine if self.engine and self.engine.ok else None,
                                       progress=lambda d, m: t.progress.emit(d, m, "Análise global"),
                                       cancelled=t.is_cancelled)
        t.progress.connect(self._prog)
        t.result.connect(lambda nb: (self.busy(False), self._restore_engine_label(), self._update_counts(),
                                     self.statusBar().showMessage("Análise global concluída.")))
        t.error.connect(lambda e: (self.busy(False), QMessageBox.critical(self, APP_NAME, e)))
        self.global_task = t
        self.tasks.append(t)
        t.start()

    def _update_counts(self):
        self._compute_counts()
        self.apply_filters()

    def _compute_counts(self):
        if not self.store:
            return
        self.settings.setValue("thr", self.sp_thr.value())
        thr = self.sp_thr.value()
        counts = {}
        for r in self.store.records:
            c = len(self.store.sha_groups.get(r.sha256, [])) - 1 if r.sha256 else 0
            nb = self.store.neighbors.get(r.id)
            if nb:
                c = max(c, sum(1 for _, s in nb if s >= thr))
            if c > 0:
                counts[r.id] = c
        self.model.counts = counts
        self.list.viewport().update()

    # ---------------- filtros
    def _rec_date(self, r):
        mode = self.cb_datefield.currentIndex()
        if mode == 0:
            return r.date_taken or r.date_file
        return r.date_taken if mode == 1 else r.date_file

    def _date_ok(self, r) -> bool:
        if not self.chk_date.isChecked():
            return True
        d = self._rec_date(r)
        if not d:
            return False
        a, b = self.de_from.date(), self.de_to.date()
        if a > b:
            a, b = b, a
        return a.toString("yyyy-MM-dd") <= d[:10] <= b.toString("yyyy-MM-dd")

    def _set_date_range_to_collection(self):
        """Ajusta 'de/até' para a menor e a maior data da coleção (sem disparar o filtro)."""
        ds = []
        if self.store and self.store.records:
            ds = [d[:10] for d in (self._rec_date(r) for r in self.store.records) if d]
        lo = QDate.fromString(min(ds), "yyyy-MM-dd") if ds else QDate.currentDate().addYears(-1)
        hi = QDate.fromString(max(ds), "yyyy-MM-dd") if ds else QDate.currentDate()
        for w_, d in ((self.de_from, lo), (self.de_to, hi)):
            w_.blockSignals(True)
            w_.setDate(d)
            w_.blockSignals(False)

    def _date_toggled(self, on):
        self.apply_filters()

    def _date_field_changed(self):
        if not self.chk_date.isChecked():
            self._set_date_range_to_collection()
        self.apply_filters()

    def _date_edited(self):
        if not self.chk_date.isChecked():
            self.chk_date.blockSignals(True)
            self.chk_date.setChecked(True)  # mexeu na data => quer filtrar
            self.chk_date.blockSignals(False)
        self.apply_filters()

    def reset_filters(self):
        """Estado padrão: todas as imagens, sem nenhum filtro."""
        widgets = (self.ed_search, self.cb_format, self.cb_special, self.chk_date, self.cb_datefield,
                   self.de_from, self.de_to, self.cb_sort, self.chk_desc)
        for w_ in widgets:
            w_.blockSignals(True)
        self._search_timer.stop()
        self.ed_search.clear()
        self.cb_format.setCurrentIndex(0)
        self.cb_special.setCurrentIndex(0)
        self.chk_date.setChecked(False)
        self.cb_datefield.setCurrentIndex(0)
        self.cb_sort.setCurrentIndex(0)
        self.chk_desc.setChecked(False)
        for w_ in widgets:
            w_.blockSignals(False)
        self._set_date_range_to_collection()
        self.apply_filters()

    def filtered(self) -> list:
        if not self.store:
            return []
        terms = self.ed_search.text().split()
        inc = [t for t in terms if not t.startswith("-")]
        exc = [t[1:] for t in terms if t.startswith("-") and len(t) > 1]
        allowed = None
        for t in inc:
            ids = self.store.text_search(t)
            allowed = ids if allowed is None else allowed & ids
        banned = set()
        for t in exc:
            banned |= self.store.text_search(t)
        fmt = self.cb_format.currentText() if self.cb_format.currentIndex() > 0 else None
        sp = self.cb_special.currentIndex()
        counts = self.model.counts
        out = []
        for r in self.store.records:
            if allowed is not None and r.id not in allowed or r.id in banned:
                continue
            if fmt and (r.fmt or "?").upper() != fmt:
                continue
            if (sp == 1 and not counts.get(r.id)) or (sp == 2 and r.gps_lat is None) or \
               (sp == 3 and r.date_taken) or (sp == 4 and not (r.emb_dist is not None and r.emb_dist > 14)) or \
               (sp == 5 and len(self.store.sha_groups.get(r.sha256, [])) < 2) or (sp == 6 and r.ext_ok) or \
               (sp == 7 and not r.error) or (sp == 8 and not r.flat):
                continue
            if not self._date_ok(r):
                continue
            out.append(r)
        k = self.cb_sort.currentIndex()
        keys = [None, lambda r: r.name.lower(), lambda r: r.date_taken or "", lambda r: r.date_file or "",
                lambda r: r.size or 0, lambda r: (r.width or 0) * (r.height or 0),
                lambda r: (r.fmt or "", r.name.lower())]
        if sp == 5:
            out.sort(key=lambda r: r.sha256 or "")
        elif k > 0:
            out.sort(key=keys[k], reverse=self.chk_desc.isChecked())
        elif self.chk_desc.isChecked():
            out.reverse()
        return out

    def apply_filters(self):
        cur = self.current.id if self.current else None
        items = self.filtered()
        self.model.set_items(items)
        n = len(self.store.records) if self.store else 0
        msg = f"{len(items)} de {n} imagens exibidas"
        if self.chk_date.isChecked() and self.store:
            sem = sum(1 for r in self.store.records if not self._rec_date(r))
            if sem:
                msg += f" · {sem} sem a data escolhida (excluídas pelo filtro de período)"
        self.statusBar().showMessage(msg)
        if cur is not None:
            for i, r in enumerate(items):
                if r.id == cur:
                    self.list.setCurrentIndex(self.model.index(i))
                    self.list.scrollTo(self.model.index(i))
                    break

    def _resize_thumbs(self, v):
        self.delegate.size = v
        self.model.layoutChanged.emit()

    def _sel_changed(self, *_):
        n = len(self.list.selectionModel().selectedIndexes())
        if n > 1:
            self.statusBar().showMessage(f"{n} imagens selecionadas — Ctrl+R gera o relatório delas")

    def selected_records(self):
        rows = sorted({i.row() for i in self.list.selectionModel().selectedIndexes()})
        return [self.model.items[r] for r in rows]

    # ---------------- seleção
    def on_current(self, idx, _prev=None):
        if not idx.isValid():
            return
        r = self.model.items[idx.row()]
        if self.current is not None and self.current.id == r.id:
            return
        self.current = r
        self._set_preview(self.model.pixmap(r))
        dt = r.date_taken or "—"
        ext = "" if r.ext_ok else " · <b style='color:#f80'>extensão divergente/ausente</b>"
        self.lbl_info.setText(
            f"<b>{r.name}</b> · {r.width}×{r.height} · {r.fmt}{ext} · {human_size(r.size or 0)}<br>"
            f"Captura EXIF: {dt} · Arquivo: {r.date_file}<br>Câmera: {r.camera or '—'}"
            + (f" · GPS: {r.gps_lat:.6f}, {r.gps_lon:.6f}" if r.gps_lat is not None else "")
            + f"<br><small>SHA-256: {r.sha256}<br>MD5: {r.md5}</small>")
        path = r.path

        def load():
            im = loader.open_pil(path, 1200)
            return path, pil_to_qimage(im)

        self.run_task(load, on_result=self._on_preview, on_error=lambda e: None)
        self._fill_meta(r)
        self._fill_emb(r)
        self.run_similar()

    def _on_preview(self, res):
        path, qi = res
        if self.current and self.current.path == path:
            self._set_preview(QPixmap.fromImage(qi))

    def _set_preview(self, pm: QPixmap):
        self._preview_pm = pm
        if pm.isNull():
            self.lbl_preview.setText("Sem prévia")
            return
        self.lbl_preview.setPixmap(pm.scaled(self.lbl_preview.size(), Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if getattr(self, "_preview_pm", None) is not None and not self._preview_pm.isNull():
            self._set_preview(self._preview_pm)

    def _fill_meta(self, r):
        meta = self.store.meta(r.id)
        self.tbl_meta.setRowCount(len(meta))
        for i, (k, v) in enumerate(meta.items()):
            self.tbl_meta.setItem(i, 0, QTableWidgetItem(k))
            it = QTableWidgetItem(str(v))
            it.setToolTip(str(v)[:2000])
            self.tbl_meta.setItem(i, 1, it)
        self._filter_meta(self.ed_meta.text())

    def _filter_meta(self, txt):
        txt = txt.lower()
        for i in range(self.tbl_meta.rowCount()):
            k = self.tbl_meta.item(i, 0).text().lower()
            v = self.tbl_meta.item(i, 1).text().lower()
            self.tbl_meta.setRowHidden(i, bool(txt) and txt not in k and txt not in v)

    def _fill_emb(self, r):
        b = self.store.emb_thumb(r.id)
        if not b:
            self.lbl_emb.clear()
            self.lbl_main_small.clear()
            self.lbl_emb_info.setText("Esta imagem não possui miniatura EXIF embutida.")
            return
        pm = bytes_to_pixmap(b)
        self.lbl_emb.setPixmap(pm.scaled(260, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.lbl_main_small.setPixmap(self.model.pixmap(r).scaled(260, 260, Qt.KeepAspectRatio,
                                                                  Qt.SmoothTransformation))
        d = r.emb_dist
        alert = (" <b style='color:#f55'>⚠ DIVERGENTE — a miniatura não corresponde à imagem principal; "
                 "possível edição posterior sem regeneração da miniatura. Verificar visualmente "
                 "(tarjas/letterbox também elevam a distância).</b>" if d is not None and d > 14
                 else " Compatível com a imagem principal.")
        self.lbl_emb_info.setText(f"Esquerda: miniatura embutida ({pm.width()}×{pm.height()}). "
                                  f"Direita: imagem principal. Distância pHash = {d}.{alert}")

    # ---------------- semelhantes
    def run_similar(self):
        if not self.current or not self.store:
            return
        if self.sim_task and self.sim_task.isRunning():
            self.sim_task.cancel()
        q = self.current
        if q.id in self._sim_cache:
            self._show_similar(q.id, self._sim_cache[q.id])
            return
        self.tree.clear()
        self.sim_progress.setVisible(True)
        self.sim_progress.setRange(0, 0)
        engine = self.engine if self.engine and self.engine.ok else None
        t = Task(lambda: None)
        t.fn = lambda: (q.id, find_similar(self.store, engine, q,
                                           progress=lambda d, n: t.progress.emit(d, n, ""),
                                           cancelled=t.is_cancelled))
        t.progress.connect(lambda d, n, _: (self.sim_progress.setRange(0, max(1, n)),
                                            self.sim_progress.setValue(d)))
        t.result.connect(self._on_similar)
        t.error.connect(lambda e: (self.sim_progress.setVisible(False),
                                   self.lbl_sim.setText(f"Erro na busca: {e}")))
        self.sim_task = t
        self.tasks = [x for x in self.tasks if x.isRunning()] + [t]
        t.start()

    def _on_similar(self, res):
        rid, results = res
        if results is None:
            return
        self._sim_cache[rid] = results
        if len(self._sim_cache) > 500:
            self._sim_cache.pop(next(iter(self._sim_cache)))
        if self.current and self.current.id == rid:
            self._show_similar(rid, results)

    def _show_similar(self, rid, results):
        self.sim_progress.setVisible(False)
        self.sim_results = results
        self.tree.clear()
        n = len(self.store.records)
        self.lbl_sim.setText(f"Comparada com todos os {n} arquivos da pasta e subpastas.")
        if self.current is not None and self.current.flat:
            self.lbl_sim.setText("<b>Imagem uniforme</b> (cor sólida / sem conteúdo): só cópias idênticas "
                                 "(SHA-256) são buscadas, para evitar falsos positivos.")
        if not results:
            self.tree.addTopLevelItem(QTreeWidgetItem(["Nenhuma imagem semelhante encontrada", "", "", "", ""]))
            self.tabs.setTabText(0, "Semelhantes (0)")
            return
        self.tabs.setTabText(0, f"Semelhantes ({len(results)})")
        self.tree.setUpdatesEnabled(False)
        self.tree.setSortingEnabled(False)
        groups: dict[str, QTreeWidgetItem] = {}
        bold = QFont()
        bold.setBold(True)
        grouped = self.chk_group.isChecked()
        for ri, r in enumerate(results):
            g = groups.get(r["cat"])
            if g is None and grouped:
                g = SimItem([r["cat"], "", "", "", ""], group_rank=len(groups))
                g.setFont(0, bold)
                g.setFirstColumnSpanned(True)
                self.tree.addTopLevelItem(g)
                groups[r["cat"]] = g
            c = r["rec"]
            relp = os.path.relpath(c.path, self.folder) if self.folder else c.path
            detail = r["detail"] if grouped else f"{r['cat']} — {r['detail']}"
            it = SimItem([f"{relp}\n{c.width}×{c.height} px · {c.fmt or ''}",
                          f"{r['score']:.0f}%", fmt_date_br(c.date_file),
                          human_size(c.size or 0) if c.size and c.size > 0 else "—", detail],
                         keys={0: relp.lower(), 1: r["score"], 2: c.date_file or None,
                               3: c.size if c.size and c.size > 0 else None, 4: detail})
            it.setToolTip(2, "Data de modificação do arquivo (sistema)"
                          + (f"\nData de captura (EXIF/XMP): {fmt_date_br(c.date_taken)}" if c.date_taken else ""))
            it.setToolTip(3, f"{c.size or 0:,} bytes".replace(",", "."))
            it.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
            it.setTextAlignment(3, Qt.AlignRight | Qt.AlignVCenter)
            it.setIcon(0, QIcon(self.model.pixmap(c)))
            it.setData(0, Qt.UserRole, ri)
            it.setToolTip(0, c.path)
            it.setToolTip(4, detail)
            if grouped:
                g.addChild(it)
            else:
                self.tree.addTopLevelItem(it)
        for g in groups.values():
            g.setText(0, f"{g.text(0)}  ({g.childCount()})")
            g.setExpanded(True)
        self.tree.setSortingEnabled(True)  # reaplica a ordenação escolhida (padrão: data crescente)
        self.tree.setUpdatesEnabled(True)

    def _group_toggled(self, on):
        self.settings.setValue("sim_group_v2", "true" if on else "false")
        if self.current is not None and self.sim_results is not None:
            self._show_similar(self.current.id, self.sim_results)

    def on_similar_open(self, item, _col):
        ri = item.data(0, Qt.UserRole)
        r = self.sim_results[ri] if isinstance(ri, int) and ri < len(self.sim_results) else None
        if r and self.current:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                dlg = CompareDialog(self.current, r, self.store, self)
            finally:
                QApplication.restoreOverrideCursor()
            dlg.exec()

    # ---------------- exportação com semelhantes
    def export_with_similar(self):
        recs = self.selected_records() or ([self.current] if self.current else [])
        if not recs or not self.store:
            QMessageBox.information(self, APP_NAME, "Selecione na galeria a(s) imagem(ns) a exportar.")
            return
        dest = QFileDialog.getExistingDirectory(
            self, f"Pasta de destino — {len(recs)} imagem(ns) + semelhantes",
            self.settings.value("export_dir", ""))
        if not dest:
            return
        src = os.path.normcase(os.path.abspath(self.store.folder))
        d = os.path.normcase(os.path.abspath(dest))
        if d == src or d.startswith(src + os.sep):
            QMessageBox.warning(self, APP_NAME, "A pasta de destino está DENTRO da pasta analisada.\n\n"
                                "Escolha outra pasta: copiar para dentro do material questionado o alteraria "
                                "e as cópias seriam indexadas na próxima varredura.")
            return
        self.settings.setValue("export_dir", dest)
        engine = self.engine if self.engine and self.engine.ok else None
        cached = {r.id: self._sim_cache[r.id] for r in recs if r.id in self._sim_cache}
        self.busy(True, "Exportando…")
        t = Task(lambda: None)

        def job():
            sets = []
            for k, r in enumerate(recs):
                if t.is_cancelled():
                    return None
                res = cached.get(r.id)
                if res is None:
                    t.progress.emit(k, len(recs), f"Buscando semelhantes de {r.name}")
                    res = find_similar(self.store, engine, r, cancelled=t.is_cancelled) or []
                sets.append((r, res))
            from .export import export_sets
            return export_sets(sets, dest, self.store.folder,
                               progress=lambda a, b, s: t.progress.emit(a, b, s), cancelled=t.is_cancelled)

        t.fn = job
        t.progress.connect(self._prog)
        t.result.connect(self._export_done)
        t.error.connect(lambda e: (self.busy(False), QMessageBox.critical(self, APP_NAME, f"Erro na exportação:\n{e}")))
        self.report_task = t
        self.tasks.append(t)
        t.start()

    def _export_done(self, summ):
        self.busy(False)
        if not summ:
            return
        mans = "\n".join(f"  {os.path.relpath(m, summ['dest'])}\n  SHA-256: {h}" for m, h in summ["manifests"])
        msg = (f"{summ['files']} arquivo(s) exportado(s) em {len(summ['dirs'])} conjunto(s):\n{summ['dest']}\n\n"
               "• Nomes originais e estrutura de subpastas preservados (nenhum arquivo renomeado ou sobrescrito).\n"
               "• Datas de criação/modificação preservadas; SHA-256 e MD5 do original (recalculados agora), "
               "da cópia e do índice conferidos.\n"
               "• Cada conjunto tem manifesto.csv, hashes_sha256.txt e hashes_md5.txt.\n\n"
               f"Manifesto(s):\n{mans}")
        if summ["bad"]:
            QMessageBox.warning(self, APP_NAME, msg + f"\n\n⚠ {summ['bad']} arquivo(s) com ocorrência "
                                "(erro, hash divergente, data não preservada ou já existente) — veja a coluna "
                                "“Conferência” do manifesto.")
        else:
            QMessageBox.information(self, APP_NAME, msg + "\n\nTodas as cópias conferem com o original e com o índice.")
        open_in_explorer(summ["dest"] if len(summ["dirs"]) != 1 else os.path.join(summ["dirs"][0], "manifesto.csv"))

    # ---------------- relatório
    def make_report(self):
        recs = self.selected_records()
        if not recs:
            QMessageBox.information(self, APP_NAME, "Selecione na galeria as imagens do relatório "
                                    "(Ctrl+clique ou Shift+clique para várias).")
            return
        base = "relatorio_similaridade_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        fn, _ = QFileDialog.getSaveFileName(self, f"Salvar relatório ({len(recs)} imagens) — gera DOCX e PDF",
                                            base, "Relatório (*.docx *.pdf)")
        if not fn:
            return
        engine = self.engine if self.engine and self.engine.ok else None
        cached = {r.id: self._sim_cache[r.id] for r in recs if r.id in self._sim_cache}
        self.busy(True, "Gerando relatório…")
        t = Task(lambda: None)
        t.fn = lambda: report.build_report(self.store, engine, recs, fn,
                                           progress=lambda d, n, s: t.progress.emit(d, n, s),
                                           cancelled=t.is_cancelled, cached=cached)
        t.progress.connect(self._prog)
        t.result.connect(self._report_done)
        t.error.connect(lambda e: (self.busy(False), QMessageBox.critical(self, APP_NAME, f"Erro no relatório:\n{e}")))
        self.report_task = t
        self.tasks.append(t)
        t.start()

    def _report_done(self, out):
        self.busy(False)
        if not out:
            return
        files = []
        if out.get("docx"):
            files.append(out["docx"])
        if out.get("html"):
            report.write_pdf(out["html"], out["pdf_target"])
            files.append(out["pdf_target"])
        msg = "Relatório gerado:\n\n" + "\n\n".join(f"{f}\nSHA-256: {report.sha256_of(f)}" for f in files)
        QMessageBox.information(self, APP_NAME, msg)

    # ---------------- exportação / outros
    def export_csv(self):
        if not self.model.items:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Exportar lista", "galeria.csv", "CSV (*.csv)")
        if not fn:
            return
        with open(fn, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Arquivo", "Caminho", "SHA-256", "MD5", "Formato detectado", "Extensão coerente",
                        "Largura", "Altura", "Bytes", "Data captura (EXIF)", "Data arquivo", "Câmera",
                        "Latitude", "Longitude", "Semelhantes (≈)", "Miniatura EXIF dist.", "Erro"])
            for r in self.model.items:
                w.writerow([r.name, r.path, r.sha256, r.md5, r.fmt, "sim" if r.ext_ok else "NÃO",
                            r.width, r.height, r.size, r.date_taken or "", r.date_file, r.camera,
                            r.gps_lat if r.gps_lat is not None else "", r.gps_lon if r.gps_lon is not None else "",
                            self.model.counts.get(r.id, 0), "" if r.emb_dist is None else r.emb_dist,
                            r.error or ""])
        self.statusBar().showMessage(f"Exportado: {fn}")

    def export_similar_csv(self):
        if not self.current or not self.sim_results:
            QMessageBox.information(self, APP_NAME, "Selecione uma imagem com semelhantes encontrados.")
            return
        base = os.path.splitext(self.current.name)[0]
        fn, _ = QFileDialog.getSaveFileName(self, "Exportar semelhantes", f"semelhantes_{base}.csv",
                                            "CSV (*.csv)")
        if not fn:
            return
        with open(fn, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Imagem de referência", self.current.path, "SHA-256", self.current.sha256])
            w.writerow(["Gerado em", datetime.now().strftime("%d/%m/%Y %H:%M:%S"), APP_NAME, __version__])
            w.writerow([])
            w.writerow(["Categoria", "Confiança (%)", "Data de modificação", "Tamanho (bytes)", "Detalhe",
                        "Arquivo", "SHA-256", "Largura", "Altura"])
            ordered = sorted(self.sim_results, key=lambda r: (r["rec"].date_file is None, r["rec"].date_file or ""))
            for r in ordered:  # mesma ordem padrão da aba: data de modificação crescente, sem data no fim
                c = r["rec"]
                w.writerow([r["cat"], f"{r['score']:.0f}", fmt_date_br(c.date_file), c.size, r["detail"],
                            c.path, c.sha256, c.width, c.height])
        self.statusBar().showMessage(f"Exportado: {fn}")

    def clear_cache(self):
        if not self.store:
            return
        if QMessageBox.question(self, APP_NAME, "Apagar o índice desta pasta? (as imagens originais não "
                                "são afetadas)") == QMessageBox.Yes:
            self.store.clear()
            self.store.load()
            self._sim_cache.clear()
            self.model.set_items([])
            self.statusBar().showMessage("Índice apagado. Reindexe a pasta (F5).")

    def about(self):
        db = self.store.db_path if self.store else "(nenhuma pasta aberta)"
        QMessageBox.about(self, APP_NAME, f"""<h3>{APP_NAME} {__version__}</h3>
<p>Galeria com busca e detecção de imagens semelhantes em toda a pasta e subpastas: cópias idênticas
(SHA-256), versões redimensionadas/recomprimidas e miniaturas (pHash), rotações/espelhamentos,
recortes e edições locais (CNN regional + ORB/RANSAC), miniatura EXIF embutida e formato real
pelo conteúdo (magic bytes).</p>
<p>{self.engine.describe() if self.engine else ''}</p>
<p><b>Somente leitura:</b> nenhum arquivo da pasta é alterado. O índice fica em<br><code>{db}</code></p>
<p>Os resultados são <i>indicativos</i> e não substituem o exame pericial.</p>""")

    def closeEvent(self, e):
        for t in self.tasks:
            if t and t.isRunning():
                t.cancel()
        for t in self.tasks:
            if t and t.isRunning():
                t.wait(5000)
        super().closeEvent(e)
