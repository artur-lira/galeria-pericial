"""Relatório (DOCX e PDF) das imagens SELECIONADAS e de suas semelhantes."""
from __future__ import annotations

import hashlib
import html
import os
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import APP_NAME, __version__, loader, similarity as sim
from .search import FEATS, find_similar

MAX_FIGS_PER_IMAGE = 15
FIG_W = 1400


def _font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _with_poly(img: Image.Image, poly, norm_w, color):
    img = img.copy()
    if poly is not None and norm_w:
        s = img.width / norm_w
        d = ImageDraw.Draw(img, "RGBA")
        pts = [(float(x) * s, float(y) * s) for x, y in poly]
        d.polygon(pts, fill=color + (45,))
        d.line(pts + [pts[0]], fill=color + (255,), width=max(3, img.width // 200))
    return img


def side_by_side(q_path, c_path, result, title_l="QUESTIONADA", title_r="SEMELHANTE") -> Image.Image:
    qi = loader.to_rgb(loader.open_pil(q_path, sim.NORM_MAX))
    ci = loader.to_rgb(loader.open_pil(c_path, sim.NORM_MAX))
    m = result.get("match")
    qf, cf = FEATS.get(q_path), FEATS.get(c_path)
    cat = result["cat"]
    if m and qf and cf:
        if cat in (sim.CAT_CONTAINS, sim.CAT_EDITED, sim.CAT_OVERLAP):
            qi = _with_poly(qi, m["poly_b_in_a"], qf.w, (0, 190, 255))
        if cat in (sim.CAT_CROP_OF, sim.CAT_OVERLAP):
            ci = _with_poly(ci, m["poly_a_in_b"], cf.w, (255, 200, 0))
    half = FIG_W // 2 - 15
    h = 0
    ims = []
    for im in (qi, ci):
        s = min(half / im.width, 600 / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        ims.append(im)
        h = max(h, im.height)
    canvas = Image.new("RGB", (FIG_W, h + 40), "white")
    d = ImageDraw.Draw(canvas)
    f = _font(20)
    for k, (im, t, p) in enumerate(zip(ims, (title_l, title_r), (q_path, c_path))):
        x = k * (FIG_W // 2) + (half - im.width) // 2
        canvas.paste(im, (x, 36))
        d.text((k * (FIG_W // 2) + 5, 6), f"{t}: {os.path.basename(p)}", fill="black", font=f)
    return canvas


def diff_figure(q_path, c_path, result) -> tuple[Image.Image, float] | None:
    cat = result["cat"]
    if cat not in (sim.CAT_EDITED, sim.CAT_NEAR, sim.CAT_THUMB, sim.CAT_CROP_OF, sim.CAT_CONTAINS):
        return None
    qi = loader.to_rgb(loader.open_pil(q_path, sim.NORM_MAX))
    ci = loader.to_rgb(loader.open_pil(c_path, sim.NORM_MAX))
    m = result.get("match")
    try:
        if cat == sim.CAT_CROP_OF and m:
            # recorte: compara a questionada com a região correspondente da outra
            arr, pct = sim.difference_map(qi, ci, m["H"])
        elif cat == sim.CAT_CONTAINS and m:
            arr, pct = sim.difference_map(ci, qi, np.linalg.inv(m["H"]))
        else:
            arr, pct = sim.difference_map(qi, ci, m["H"] if m else None)
    except Exception:  # noqa: BLE001
        return None
    return Image.fromarray(arr), pct


def _fields(store, rec) -> list[tuple[str, str]]:
    meta = store.meta(rec.id)
    gps = f"{rec.gps_lat:.6f}, {rec.gps_lon:.6f}" if rec.gps_lat is not None else "—"
    return [
        ("Arquivo", rec.name), ("Caminho", rec.path),
        ("Tamanho", f"{rec.size:,} bytes".replace(",", ".")),
        ("Formato detectado (assinatura)", rec.fmt or "—"),
        ("Extensão x conteúdo", meta.get("Extensão x conteúdo", "—")),
        ("Dimensões", f"{rec.width} x {rec.height} px"),
        ("Data de captura (EXIF)", rec.date_taken or "—"),
        ("Data de modificação (sistema)", rec.date_file or "—"),
        ("Câmera / dispositivo", rec.camera or "—"),
        ("Software (EXIF)", meta.get("EXIF: Software", "—")),
        ("GPS", gps),
        ("Miniatura EXIF", meta.get("Miniatura EXIF embutida", "ausente")),
        ("SHA-256", rec.sha256 or "—"), ("MD5", rec.md5 or "—"),
    ]


METHOD = [
    "Cópias idênticas: comparação do hash SHA-256 do conteúdo integral dos arquivos.",
    "Versões redimensionadas, recomprimidas, miniaturas e caches: hash perceptual pHash (DCT 32x32, "
    "64 bits), calculado também para as 8 orientações (rotações de 90° e espelhamentos); limiar de "
    f"distância de Hamming ≤ {sim.PHASH_NEAR}.",
    "Triagem de recortes e edições em toda a coleção: descritores regionais de rede neural convolucional "
    "(ResNet-50/ImageNet, camada 3, agregação GeM em 15 regiões/escala), comparados por similaridade de "
    "cosseno contra todos os arquivos da pasta e subpastas.",
    "Confirmação geométrica: pontos-chave ORB, casamento com teste de razão e estimativa de homografia por "
    f"RANSAC (mínimo de {sim.MIN_INLIERS} correspondências consistentes). A homografia localiza o recorte "
    "na imagem maior e permite alinhar as imagens.",
    "Alterações locais: diferença absoluta entre as imagens alinhadas, após normalização global de brilho e "
    "contraste, com limiarização e morfologia matemática.",
    "Formato dos arquivos determinado pela assinatura binária (magic bytes), independentemente da extensão.",
    "Todos os arquivos foram acessados somente para leitura.",
]

LIMITS = [
    "Os resultados automatizados têm caráter de triagem e devem ser confirmados por exame visual do perito.",
    "Recortes muito pequenos (abaixo de ~5% da área) ou de regiões sem textura podem não ser detectados.",
    "Os percentuais de confiança são heurísticos (baseados no número de correspondências) e não "
    "representam probabilidade estatística.",
    "A miniatura EXIF divergente é indício, não prova, de edição: tarjas (letterbox) também elevam a distância.",
]


def build_report(store, engine, recs, out_path: str, want_docx=True, want_pdf=True,
                 progress=None, cancelled=lambda: False, cached: dict | None = None):
    """Gera o conteúdo do relatório. Retorna dict com os dados para o PDF (feito pela UI) e o DOCX salvo."""
    tmp = tempfile.mkdtemp(prefix="galeria_rel_")
    sections = []
    n = len(recs)
    for k, rec in enumerate(recs):
        if cancelled():
            return None
        if progress:
            progress(k, n, f"Analisando {rec.name}")
        res = (cached or {}).get(rec.id)
        if res is None:
            res = find_similar(store, engine, rec, cancelled=cancelled) or []
        th = os.path.join(tmp, f"q{k}.jpg")
        try:
            im = loader.to_rgb(loader.open_pil(rec.path, 700))
            im.save(th, quality=90)
        except Exception:  # noqa: BLE001
            th = None
        figs = []
        for i, r in enumerate(res[:MAX_FIGS_PER_IMAGE]):
            if cancelled():
                return None
            if progress:
                progress(k, n, f"Figuras de {rec.name}: {i + 1}/{min(len(res), MAX_FIGS_PER_IMAGE)}")
            try:
                sb = side_by_side(rec.path, r["path"], r)
                p1 = os.path.join(tmp, f"f{k}_{i}_a.jpg")
                sb.save(p1, quality=88)
                df = diff_figure(rec.path, r["path"], r)
                p2, pct = None, None
                if df:
                    p2 = os.path.join(tmp, f"f{k}_{i}_d.jpg")
                    w = min(FIG_W, df[0].width * 2)
                    df[0].resize((w, int(df[0].height * w / df[0].width))).save(p2, quality=88)
                    pct = df[1]
                figs.append({"r": r, "side": p1, "diff": p2, "pct": pct, "n": i + 1})
            except Exception as e:  # noqa: BLE001
                figs.append({"r": r, "side": None, "diff": None, "pct": None, "n": i + 1, "err": str(e)})
        sections.append({"rec": rec, "fields": _fields(store, rec), "thumb": th, "results": res,
                         "figs": figs})
    info = {
        "title": "RELATÓRIO TÉCNICO DE ANÁLISE DE SIMILARIDADE DE IMAGENS",
        "generated": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "software": f"{APP_NAME} {__version__}",
        "folder": store.folder,
        "total": len(store.records),
        "last_scan": store.get_meta("last_scan") or "—",
        "device": engine.describe() if engine else "CNN desativada",
        "sections": sections,
    }
    out = {"info": info}
    if want_docx:
        if progress:
            progress(n, n, "Gravando DOCX…")
        p = os.path.splitext(out_path)[0] + ".docx"
        write_docx(info, p)
        out["docx"] = p
    if want_pdf:
        out["pdf_target"] = os.path.splitext(out_path)[0] + ".pdf"
        out["html"] = to_html(info)
    return out


def rel(path, folder):
    try:
        return os.path.relpath(path, folder)
    except ValueError:
        return path


# ---------------------------------------------------------------------------- DOCX
def write_docx(info, path):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt

    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2)
    sec.top_margin = sec.bottom_margin = Cm(2)
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(10)
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = t.add_run(info["title"])
    run.bold = True
    run.font.size = Pt(14)

    def table(rows, header=None, widths=None, size=9):
        tb = doc.add_table(rows=0, cols=len(rows[0]) if rows else len(header))
        tb.style = "Table Grid"
        if header:
            cells = tb.add_row().cells
            for c, h in zip(cells, header):
                c.text = ""
                r = c.paragraphs[0].add_run(str(h))
                r.bold = True
                r.font.size = Pt(size)
        for row in rows:
            cells = tb.add_row().cells
            for c, v in zip(cells, row):
                c.text = ""
                r = c.paragraphs[0].add_run(str(v))
                r.font.size = Pt(size)
        if widths:
            for row in tb.rows:
                for c, w in zip(row.cells, widths):
                    c.width = Cm(w)
        return tb

    doc.add_heading("1. Identificação", level=1)
    table([("Gerado em", info["generated"]), ("Software", info["software"]),
           ("Pasta analisada (com subpastas)", info["folder"]),
           ("Arquivos de imagem indexados", str(info["total"])),
           ("Última indexação", info["last_scan"]), ("Processamento", info["device"]),
           ("Imagens questionadas neste relatório", str(len(info["sections"]))),
           ("Perito responsável", ""), ("Matrícula", ""), ("Requisição / Procedimento", "")],
          widths=[6, 11])
    doc.add_heading("2. Metodologia", level=1)
    for m in METHOD:
        doc.add_paragraph(m, style="List Bullet")

    doc.add_heading("3. Imagens questionadas e semelhantes", level=1)
    for k, s in enumerate(info["sections"], 1):
        rec = s["rec"]
        doc.add_heading(f"3.{k} {rec.name}", level=2)
        if s["thumb"]:
            doc.add_picture(s["thumb"], width=Cm(8))
        table(s["fields"], widths=[5, 12])
        doc.add_paragraph()
        p = doc.add_paragraph()
        p.add_run(f"Semelhantes encontradas em toda a pasta: {len(s['results'])}").bold = True
        if s["results"]:
            rows = []
            for i, r in enumerate(s["results"], 1):
                c = r["rec"]
                rows.append((i, r["cat"], rel(c.path, info["folder"]), f"{c.width}x{c.height}",
                             f"{r['score']:.0f}%", r["detail"], c.sha256 or ""))
            table(rows, header=["#", "Relação", "Arquivo", "Dimensões", "Conf.", "Detalhe", "SHA-256"],
                  widths=[0.8, 3, 3.5, 1.8, 1.2, 3.5, 3.2], size=7)
        for f in s["figs"]:
            r = f["r"]
            doc.add_paragraph()
            cap = doc.add_paragraph()
            cap.add_run(f"Figura {k}.{f['n']} — {r['cat']}: {rel(r['path'], info['folder'])}").bold = True
            if f.get("side"):
                doc.add_picture(f["side"], width=Cm(16.5))
                doc.add_paragraph("À esquerda, a imagem questionada; à direita, a semelhante. A área "
                                  "correspondente aparece destacada quando há recorte.").runs[0].font.size = Pt(8)
            if f.get("diff"):
                doc.add_picture(f["diff"], width=Cm(12))
                doc.add_paragraph(f"Mapa de diferenças após alinhamento: em vermelho/caixas amarelas, as áreas "
                                  f"divergentes ({f['pct']:.2f}% da área comum). Áreas escurecidas não têm "
                                  f"correspondência.").runs[0].font.size = Pt(8)
            if f.get("err"):
                doc.add_paragraph(f"(figura não gerada: {f['err']})")
    doc.add_heading("4. Limitações e observações", level=1)
    for m in LIMITS:
        doc.add_paragraph(m, style="List Bullet")
    doc.add_heading("5. Relação de hashes", level=1)
    seen, rows = set(), []
    for s in info["sections"]:
        for c in [s["rec"]] + [r["rec"] for r in s["results"]]:
            if c.path not in seen:
                seen.add(c.path)
                rows.append((rel(c.path, info["folder"]), c.sha256 or "", c.md5 or ""))
    table(rows, header=["Arquivo", "SHA-256", "MD5"], widths=[6, 7, 4], size=7)
    doc.save(path)


# ---------------------------------------------------------------------------- HTML (PDF)
def to_html(info) -> str:
    e = html.escape
    def img(p, w):
        return f'<p><img src="{e(p.replace(chr(92), "/"))}" width="{w}"></p>' if p else ""

    def tbl(rows, header=None, fs=9):
        h = "<table border='1' cellspacing='0' cellpadding='3' width='100%' style='border-collapse:collapse;" \
            f"font-size:{fs}pt'>"
        if header:
            h += "<tr>" + "".join(f"<th bgcolor='#dddddd'>{e(str(x))}</th>" for x in header) + "</tr>"
        for r in rows:
            h += "<tr>" + "".join(f"<td>{e(str(x))}</td>" for x in r) + "</tr>"
        return h + "</table>"

    o = [f"<html><body style='font-family:Arial;font-size:10pt'><h2 align='center'>{e(info['title'])}</h2>"]
    o.append("<h3>1. Identificação</h3>")
    o.append(tbl([("Gerado em", info["generated"]), ("Software", info["software"]),
                  ("Pasta analisada (com subpastas)", info["folder"]),
                  ("Arquivos de imagem indexados", info["total"]), ("Última indexação", info["last_scan"]),
                  ("Processamento", info["device"]),
                  ("Imagens questionadas neste relatório", len(info["sections"])),
                  ("Perito responsável", ""), ("Matrícula", ""), ("Requisição / Procedimento", "")]))
    o.append("<h3>2. Metodologia</h3><ul>" + "".join(f"<li>{e(m)}</li>" for m in METHOD) + "</ul>")
    o.append("<h3>3. Imagens questionadas e semelhantes</h3>")
    for k, s in enumerate(info["sections"], 1):
        rec = s["rec"]
        o.append(f"<h4>3.{k} {e(rec.name)}</h4>")
        o.append(img(s["thumb"], 300))
        o.append(tbl(s["fields"]))
        o.append(f"<p><b>Semelhantes encontradas em toda a pasta: {len(s['results'])}</b></p>")
        if s["results"]:
            rows = [(i, r["cat"], rel(r["path"], info["folder"]), f"{r['rec'].width}x{r['rec'].height}",
                     f"{r['score']:.0f}%", r["detail"], r["rec"].sha256 or "")
                    for i, r in enumerate(s["results"], 1)]
            o.append(tbl(rows, ["#", "Relação", "Arquivo", "Dimensões", "Conf.", "Detalhe", "SHA-256"], 7))
        for f in s["figs"]:
            r = f["r"]
            o.append(f"<p><b>Figura {k}.{f['n']} — {e(r['cat'])}: {e(rel(r['path'], info['folder']))}</b></p>")
            if f.get("side"):
                o.append(img(f["side"], 640))
            if f.get("diff"):
                o.append(img(f["diff"], 460))
                o.append(f"<p style='font-size:8pt'>Mapa de diferenças após alinhamento: "
                         f"{f['pct']:.2f}% da área comum divergente.</p>")
    o.append("<h3>4. Limitações e observações</h3><ul>" + "".join(f"<li>{e(m)}</li>" for m in LIMITS) + "</ul>")
    seen, rows = set(), []
    for s in info["sections"]:
        for c in [s["rec"]] + [r["rec"] for r in s["results"]]:
            if c.path not in seen:
                seen.add(c.path)
                rows.append((rel(c.path, info["folder"]), c.sha256 or "", c.md5 or ""))
    o.append("<h3>5. Relação de hashes</h3>" + tbl(rows, ["Arquivo", "SHA-256", "MD5"], 7))
    o.append("</body></html>")
    return "".join(o)


def write_pdf(html_text: str, path: str):
    """Precisa rodar na thread da interface (Qt)."""
    from PySide6.QtCore import QMarginsF, QSizeF
    from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter, QTextDocument

    w = QPdfWriter(path)
    w.setPageSize(QPageSize(QPageSize.A4))
    w.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Millimeter)
    w.setResolution(150)
    w.setTitle("Relatório de similaridade de imagens")
    w.setCreator(f"{APP_NAME} {__version__}")
    doc = QTextDocument()
    doc.setHtml(html_text)
    doc.setPageSize(QSizeF(w.width(), w.height()))
    doc.print_(w)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
