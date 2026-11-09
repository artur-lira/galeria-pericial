"""Identificação do formato pela assinatura do arquivo ("magic bytes").

Serve para os arquivos sem extensão, com extensão desconhecida e com extensão
trocada (ex.: um PNG renomeado para .jpg ou .dat). O formato detectado
prevalece sobre a extensão.
"""
from __future__ import annotations

import os

HEADER_LEN = 64

# extensões que apontam o formato esperado (para acusar divergência)
EXT_TO_FMT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".jpe": "JPEG", ".jfif": "JPEG",
    ".png": "PNG", ".gif": "GIF", ".bmp": "BMP", ".dib": "BMP",
    ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP", ".ico": "ICO", ".cur": "ICO",
    ".psd": "PSD", ".heic": "HEIF", ".heif": "HEIF", ".avif": "AVIF",
    ".jp2": "JPEG2000", ".j2k": "JPEG2000", ".jpx": "JPEG2000", ".jxl": "JXL",
    ".dds": "DDS", ".icns": "ICNS", ".exr": "EXR", ".hdr": "HDR", ".pcx": "PCX",
    ".ppm": "PNM", ".pgm": "PNM", ".pbm": "PNM", ".pnm": "PNM", ".eps": "EPS",
    ".tga": "TGA", ".cr3": "RAW", ".cr2": "RAW", ".nef": "RAW", ".nrw": "RAW", ".arw": "RAW",
    ".srf": "RAW", ".sr2": "RAW", ".dng": "RAW", ".orf": "RAW", ".rw2": "RAW", ".raf": "RAW",
    ".pef": "RAW", ".srw": "RAW", ".x3f": "RAW", ".3fr": "RAW", ".erf": "RAW", ".kdc": "RAW",
    ".mrw": "RAW", ".mef": "RAW", ".iiq": "RAW", ".rwl": "RAW", ".crw": "RAW",
}
# formatos TIFF-base aceitos sob extensões RAW
TIFF_LIKE = {"TIFF", "RAW"}

HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"mif2"}
AVIF_BRANDS = {b"avif", b"avis"}


def sniff_bytes(h: bytes) -> str | None:
    """Retorna o nome do formato a partir do cabeçalho, ou None se não for imagem."""
    if len(h) < 4:
        return None
    if h[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if h[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if h[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if h[:2] == b"BM" and len(h) >= 18 and int.from_bytes(h[14:18], "little") in (12, 40, 52, 56, 64, 108, 124):
        return "BMP"
    if h[:4] == b"RIFF" and h[8:12] == b"WEBP":
        return "WEBP"
    if h[:4] in (b"II*\x00", b"MM\x00*"):
        if h[8:10] == b"CR":
            return "RAW"  # Canon CR2
        return "TIFF"
    if h[:4] in (b"IIRO", b"IIRS", b"IIU\x00", b"II+\x00", b"MM\x00+"):
        return "RAW"  # Olympus ORF, Panasonic RW2, BigTIFF
    if h[:15] == b"FUJIFILMCCD-RAW":
        return "RAW"
    if h[:4] == b"FOVb":
        return "RAW"  # Sigma X3F
    if h[4:8] == b"ftyp":
        brand = h[8:12]
        brands = {h[i:i + 4] for i in range(8, min(len(h), 40), 4)}
        if brand == b"crx " or b"crx " in brands:
            return "RAW"  # Canon CR3
        if brand in AVIF_BRANDS:
            return "AVIF"
        if brand in HEIF_BRANDS or brands & HEIF_BRANDS:
            return "HEIF"
        if brands & AVIF_BRANDS:
            return "AVIF"
        return None  # MP4/MOV etc.
    if h[:12] == b"\x00\x00\x00\x0cjP  \r\n\x87\n" or h[:4] == b"\xff\x4f\xff\x51":
        return "JPEG2000"
    if h[:2] == b"\xff\x0a" or h[:12] == b"\x00\x00\x00\x0cJXL \r\n\x87\n":
        return "JXL"
    if h[:4] == b"8BPS":
        return "PSD"
    if h[:4] == b"\x00\x00\x01\x00" and 0 < int.from_bytes(h[4:6], "little") < 256:
        return "ICO"
    if h[:4] == b"\x00\x00\x02\x00" and 0 < int.from_bytes(h[4:6], "little") < 256:
        return "ICO"  # cursor
    if h[:4] == b"DDS ":
        return "DDS"
    if h[:4] == b"icns":
        return "ICNS"
    if h[:4] == b"\x76\x2f\x31\x01":
        return "EXR"
    if h[:10] == b"#?RADIANCE" or h[:6] == b"#?RGBE":
        return "HDR"
    if h[:4] == b"\xc5\xd0\xd3\xc6" or (h[:4] == b"%!PS" and b"EPSF" in h[:40]):
        return "EPS"
    if h[0] == 0x0A and h[1] in (0, 2, 3, 4, 5) and h[2] == 1:
        return "PCX"
    if h[0:1] == b"P" and h[1:2] in b"123456" and h[2:3] in (b"\n", b" ", b"\r", b"\t"):
        return "PNM"
    if h[:4] == b"\x59\xa6\x6a\x95":
        return "SUN"
    if h[:2] == b"\x01\xda":
        return "SGI"
    return None


def sniff_file(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return sniff_bytes(f.read(HEADER_LEN))
    except OSError:
        return None


def ext_matches(path: str, fmt: str | None) -> bool:
    """True se a extensão é coerente com o formato detectado (ou não há como saber)."""
    ext = os.path.splitext(path)[1].lower()
    exp = EXT_TO_FMT.get(ext)
    if fmt is None or exp is None:
        return exp is not None or fmt is None and ext == ""
    if exp == fmt:
        return True
    if exp == "RAW" and fmt in TIFF_LIKE:
        return True
    if exp == "HEIF" and fmt == "AVIF" or exp == "AVIF" and fmt == "HEIF":
        return True
    return False


def describe_ext_status(path: str, fmt: str | None) -> str:
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        return f"sem extensão (detectado: {fmt})"
    if ext not in EXT_TO_FMT:
        return f"extensão '{ext}' não é de imagem (detectado: {fmt})"
    if not ext_matches(path, fmt):
        return f"extensão '{ext}' DIVERGE do conteúdo (detectado: {fmt})"
    return "coerente"
