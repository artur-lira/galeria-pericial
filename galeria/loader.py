"""Carregamento de imagens em múltiplos formatos e extração de metadados.

Todas as leituras são SOMENTE LEITURA: nenhum arquivo da pasta analisada é
alterado, renomeado ou recebe arquivos auxiliares (o cache fica no perfil do
usuário). Isso preserva a integridade do material questionado.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps, ExifTags

from .formats import sniff_bytes, sniff_file, describe_ext_status

Image.MAX_IMAGE_PIXELS = None  # imagens periciais podem ser muito grandes

# ---- Suportes opcionais -------------------------------------------------
try:  # HEIC/HEIF/AVIF
    import pillow_heif

    pillow_heif.register_heif_opener()
    try:
        pillow_heif.register_avif_opener()
    except Exception:
        pass
    HAS_HEIF = True
except Exception:
    HAS_HEIF = False

try:  # RAW de câmeras
    import rawpy

    HAS_RAW = True
except Exception:
    HAS_RAW = False

try:
    import cv2

    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

RAW_EXT = {".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".dng",
           ".orf", ".rw2", ".raf", ".pef", ".srw", ".x3f", ".3fr", ".erf", ".kdc",
           ".mrw", ".mef", ".iiq", ".rwl"}
STD_EXT = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".dib", ".gif",
           ".tif", ".tiff", ".webp", ".ico", ".cur", ".psd", ".tga", ".ppm",
           ".pgm", ".pbm", ".pnm", ".pcx", ".dds", ".jp2", ".j2k", ".jpx",
           ".icns", ".sgi", ".im", ".msp", ".xbm", ".eps", ".emf", ".wmf",
           ".heic", ".heif", ".avif", ".jxl", ".hdr", ".exr", ".pic", ".sr", ".ras"}
IMAGE_EXT = STD_EXT | RAW_EXT

EXIF_TAGS = {**ExifTags.TAGS}
GPS_TAGS = {**ExifTags.GPSTAGS}


def is_image_file(path: str) -> bool:
    """Pela extensão (atalho). Arquivos sem extensão/desconhecidos são checados por assinatura."""
    return os.path.splitext(path)[1].lower() in IMAGE_EXT


def _source(path: str, data: bytes | None):
    return io.BytesIO(data) if data is not None else path


def open_pil(path: str, max_side: int | None = None, data: bytes | None = None,
             fmt: str | None = None) -> Image.Image:
    """Abre qualquer formato suportado (detectado pelo CONTEÚDO, não pela extensão).

    max_side: se informado, decodifica já reduzido (JPEG usa 'draft', muito mais rápido).
    data: conteúdo do arquivo já lido (evita segunda leitura do disco).
    """
    if fmt is None:
        fmt = sniff_bytes(data[:64]) if data is not None else sniff_file(path)
    ext = os.path.splitext(path)[1].lower()
    is_raw = fmt == "RAW" or (fmt in (None, "TIFF") and ext in RAW_EXT)
    if is_raw and HAS_RAW:
        try:
            src = io.BytesIO(data) if data is not None else path
            with rawpy.imread(src) as raw:
                try:  # a prévia JPEG embutida é muito mais rápida
                    if max_side:
                        th = raw.extract_thumb()
                        if th.format == rawpy.ThumbFormat.JPEG:
                            img = Image.open(io.BytesIO(th.data))
                            w0, h0 = img.size
                            k = max_side / max(w0, h0)
                            img.draft("RGB", (max(1, int(w0 * k)), max(1, int(h0 * k))))
                            img = ImageOps.exif_transpose(img).convert("RGB")
                            img.thumbnail((max_side, max_side))
                            return img
                except Exception:
                    pass
                rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            img = Image.fromarray(rgb)
            if max_side:
                img.thumbnail((max_side, max_side))
            return img
        except Exception:
            if fmt == "RAW" and not HAS_RAW:
                raise
    try:
        img = Image.open(_source(path, data))
        if max_side and img.format == "JPEG":
            # o draft só reduz enquanto AMBOS os lados ficarem >= ao pedido; pedir na proporção
            # da imagem permite usar a escala 1/8 (ex.: 4000x3000 -> 500x375 em vez de 1000x750)
            w0, h0 = img.size
            k = max_side / max(w0, h0)
            img.draft("RGB", (max(1, int(w0 * k)), max(1, int(h0 * k))))
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)  # GIF/TIFF multipágina: 1º quadro
        img = ImageOps.exif_transpose(img)  # respeita a orientação EXIF
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.getbands() or img.mode == "P" else "RGB")
        if max_side:
            img.thumbnail((max_side, max_side), Image.BILINEAR)
        return img
    except Exception:
        if HAS_CV2:  # fallback: OpenCV (EXR, HDR, alguns TIFF exóticos)
            buf = np.frombuffer(data, np.uint8) if data is not None else np.fromfile(path, dtype=np.uint8)
            arr = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            if arr is not None:
                if arr.dtype != np.uint8:
                    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                if max_side:
                    img.thumbnail((max_side, max_side))
                return img
        if is_raw and not HAS_RAW:
            raise RuntimeError("Formato RAW: instale o pacote 'rawpy'")
        raise


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB") if img.mode != "RGB" else img


def hashes_bytes(data: bytes) -> tuple[str, str]:
    """(SHA-256, MD5) do conteúdo."""
    return hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---- Metadados ----------------------------------------------------------
def _fmt_value(v) -> str:
    if isinstance(v, bytes):
        try:
            s = v.decode("utf-16-le" if v[1:2] == b"\x00" else "utf-8").strip("\x00 ")
            if s and s.isprintable():
                return s
        except Exception:
            pass
        return f"<{len(v)} bytes>"
    if isinstance(v, tuple):
        return ", ".join(_fmt_value(x) for x in v)
    try:
        if hasattr(v, "numerator") and hasattr(v, "denominator") and v.denominator not in (0, 1):
            return f"{float(v):.4g}"
    except Exception:
        pass
    return str(v).strip("\x00 ")


def _gps_to_decimal(gps: dict) -> tuple[float, float] | None:
    try:
        def conv(vals, ref):
            d, m, s = (float(x) for x in vals)
            dec = d + m / 60 + s / 3600
            return -dec if ref in ("S", "W") else dec

        lat = conv(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        lon = conv(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        return lat, lon
    except Exception:
        return None


_DATE_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y:%m:%dT%H:%M:%S",
                 "%Y/%m/%d %H:%M:%S", "%Y:%m:%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M",
                 "%Y:%m:%d", "%Y-%m-%d")


def _parse_exif_date(s: str) -> str | None:
    """Aceita os formatos usuais de EXIF/XMP/PNG. Datas zeradas ('0000:00:00') são ignoradas."""
    s = (s or "").strip().strip("\x00").strip()
    if not s or s.startswith("0000") or s.startswith("    "):
        return None
    for cand in (s[:19], s[:16], s[:10]):
        for fmt in _DATE_FORMATS:
            try:
                d = datetime.strptime(cand, fmt)
                if 1900 <= d.year <= 2100:
                    return d.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
    for fmt in ("%a, %d %b %Y %H:%M:%S", "%d %b %Y %H:%M:%S"):  # "Creation Time" de PNG (RFC 1123)
        try:
            return datetime.strptime(s[:31].replace(" GMT", "").strip(), fmt).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
    return None


_XMP_DATE = re.compile(r"(DateTimeOriginal|CreateDate|DateCreated|DateTimeDigitized)"
                       r"(?:>|=\s*\")\s*(\d{4}[-:]\d{2}[-:]\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?)")


def _fallback_date(meta: dict) -> tuple[str, str] | None:
    """Data de captura fora do EXIF: XMP, chunks de texto PNG etc. Retorna (data, origem)."""
    xmp = meta.get("XMP", "")
    for m in _XMP_DATE.finditer(xmp):
        d = _parse_exif_date(m.group(2))
        if d:
            return d, f"XMP {m.group(1)}"
    for k, v in meta.items():
        kl = k.lower()
        if kl.startswith("info: ") and any(x in kl for x in ("creation time", "date:create", "datetime",
                                                               "date created", "createdate")):
            d = _parse_exif_date(str(v))
            if d:
                return d, k
    return None


def extract_metadata(path: str, data: bytes | None = None, fmt: str | None = None) -> dict:
    """Retorna dict com metadados do arquivo e da imagem (EXIF, GPS, PNG text, XMP)."""
    st = os.stat(path)
    meta: dict[str, str] = {
        "Arquivo": os.path.basename(path),
        "Caminho": path,
        "Tamanho (bytes)": str(st.st_size),
        "Modificado (sistema)": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "Criado/alterado (sistema)": datetime.fromtimestamp(st.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "Formato detectado (assinatura)": fmt or "desconhecido",
        "Extensão x conteúdo": describe_ext_status(path, fmt),
    }
    info = {"width": 0, "height": 0, "format": fmt or os.path.splitext(path)[1].lstrip(".").upper(),
            "date_taken": None, "camera": "", "gps": None}
    ext = os.path.splitext(path)[1].lower()
    try:
        if (fmt == "RAW" or ext in RAW_EXT) and HAS_RAW:
            try:
                with rawpy.imread(io.BytesIO(data) if data is not None else path) as raw:
                    s = raw.sizes
                    info["width"], info["height"] = s.width, s.height
            except Exception:
                pass
        with Image.open(_source(path, data)) as img:
            info["width"], info["height"] = img.size
            info["format"] = fmt if fmt == "RAW" else (img.format or info["format"])
            meta["Formato"] = f"{img.format} ({img.format_description})" if img.format else info["format"]
            meta["Modo de cor"] = img.mode
            meta["Dimensões"] = f"{img.width} x {img.height}"
            if getattr(img, "n_frames", 1) > 1:
                meta["Quadros/páginas"] = str(img.n_frames)
            for k, v in (img.info or {}).items():
                if k in ("exif", "icc_profile", "xmp", "photoshop", "XML:com.adobe.xmp"):
                    if k == "icc_profile":
                        meta["Perfil ICC"] = f"<{len(v)} bytes>"
                    elif k in ("xmp", "XML:com.adobe.xmp"):
                        x = v.decode("utf-8", "ignore") if isinstance(v, bytes) else str(v)
                        meta["XMP"] = x[:4000]
                    continue
                meta[f"Info: {k}"] = _fmt_value(v)[:500]
            exif = img.getexif()
            if exif:
                all_tags = dict(exif.items())
                for ifd_id in (ExifTags.IFD.Exif, ExifTags.IFD.Interop):
                    try:
                        all_tags.update(exif.get_ifd(ifd_id))
                    except Exception:
                        pass
                for tag, val in all_tags.items():
                    name = EXIF_TAGS.get(tag, f"Tag 0x{tag:04X}")
                    if name in ("ExifOffset", "GPSInfo", "InteropOffset", "MakerNote"):
                        if name == "MakerNote":
                            meta["EXIF: MakerNote"] = f"<{len(val) if hasattr(val, '__len__') else '?'} bytes>"
                        continue
                    meta[f"EXIF: {name}"] = _fmt_value(val)[:500]
                try:
                    gps_raw = exif.get_ifd(ExifTags.IFD.GPSInfo)
                    gps = {GPS_TAGS.get(k, str(k)): v for k, v in gps_raw.items()}
                    for k, v in gps.items():
                        meta[f"GPS: {k}"] = _fmt_value(v)
                    dec = _gps_to_decimal(gps)
                    if dec:
                        info["gps"] = dec
                        meta["GPS: Decimal"] = f"{dec[0]:.6f}, {dec[1]:.6f}"
                except Exception:
                    pass
                for key in ("EXIF: DateTimeOriginal", "EXIF: DateTimeDigitized", "EXIF: DateTime"):
                    if key in meta:
                        d = _parse_exif_date(meta[key])
                        if d:
                            info["date_taken"] = d
                            meta["Data de captura (origem)"] = key
                            break
                info["camera"] = " ".join(x for x in (meta.get("EXIF: Make", ""),
                                                      meta.get("EXIF: Model", "")) if x).strip()
    except Exception as e:  # noqa: BLE001
        meta["Erro de leitura"] = str(e)
    if not info["date_taken"]:
        fb = _fallback_date(meta)
        if fb:
            info["date_taken"] = fb[0]
            meta["Data de captura (origem)"] = fb[1]
    return {"meta": meta, **info}


# ---- Miniatura EXIF embutida -------------------------------------------
def extract_embedded_thumbnail(path: str, data: bytes | None = None) -> Image.Image | None:
    """Extrai a miniatura JPEG embutida no bloco EXIF (APP1) de JPEG/TIFF/RAW.

    Útil pericialmente: se a foto foi editada e a miniatura não foi
    regenerada, a miniatura pode mostrar o conteúdo ORIGINAL.
    """
    try:
        with Image.open(_source(path, data)) as img:
            exif = img.getexif()
            # IFD1 contém JPEGInterchangeFormat (0x0201) e Length (0x0202)
            try:
                ifd1 = exif.get_ifd(ExifTags.IFD.IFD1)
            except Exception:
                ifd1 = {}
            raw_exif = img.info.get("exif")
            if ifd1 and raw_exif and 0x0201 in ifd1 and 0x0202 in ifd1:
                tiff = raw_exif[6:] if raw_exif.startswith(b"Exif\x00\x00") else raw_exif
                off, ln = int(ifd1[0x0201]), int(ifd1[0x0202])
                data = tiff[off:off + ln]
                if data[:2] == b"\xff\xd8":
                    t = Image.open(io.BytesIO(data))
                    t.load()
                    return t.convert("RGB")
    except Exception:
        pass
    # Varredura bruta: procura um JPEG dentro dos primeiros 256 KB (APP1)
    try:
        if data is not None:
            head = data[:262144]
        else:
            with open(path, "rb") as f:
                head = f.read(262144)
        start = 0
        if head[:2] == b"\xff\xd8":
            start = 2  # pula o SOI da imagem principal
        i = head.find(b"\xff\xd8\xff", start)
        while i != -1:
            j = head.find(b"\xff\xd9", i)
            if j == -1:
                break
            try:
                t = Image.open(io.BytesIO(head[i:j + 2]))
                t.load()
                if 32 <= t.width <= 1024 and 32 <= t.height <= 1024:
                    return t.convert("RGB")
            except Exception:
                pass
            i = head.find(b"\xff\xd8\xff", i + 3)
    except Exception:
        pass
    return None
