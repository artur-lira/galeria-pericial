"""Exportação da(s) imagem(ns) selecionada(s) e de suas semelhantes para uma pasta.

Requisitos periciais:
- NOME ORIGINAL PRESERVADO: os arquivos nunca são renomeados. Ordem e relação ficam só no manifesto.
- ESTRUTURA PRESERVADA: o caminho relativo à pasta analisada é reproduzido no destino (evita
  conflito entre arquivos homônimos de subpastas diferentes e mantém a proveniência visível).
- DATAS PRESERVADAS: modificação e acesso (copy2) e, no Windows, também a data de CRIAÇÃO.
- NADA É SOBRESCRITO: se o destino já existe, o arquivo não é copiado (e o fato é registrado).
- INTEGRIDADE: SHA-256/MD5 do original recalculados no momento da exportação, comparados com o
  índice e com a cópia; resultado em manifesto.csv e em hashes_sha256.txt / hashes_md5.txt
  (formato padrão, conferível com certutil, Get-FileHash ou sha256sum).
- Os originais são apenas lidos.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import sys
from datetime import datetime

from . import APP_NAME, __version__, similarity as sim

IS_WIN = os.name == "nt"


# ---------------------------------------------------------------------------- utilidades
def _long(p: str) -> str:
    """Caminho longo no Windows (> 260 caracteres) sem alterar o caminho lógico."""
    if IS_WIN:
        p = os.path.abspath(p)
        if not p.startswith("\\\\?\\"):
            return "\\\\?\\UNC\\" + p[2:] if p.startswith("\\\\") else "\\\\?\\" + p
    return p


def _hashes(path: str) -> tuple[str, str]:
    s, m = hashlib.sha256(), hashlib.md5()
    with open(_long(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            s.update(b)
            m.update(b)
    return s.hexdigest(), m.hexdigest()


def _birthtime(st) -> float | None:
    bt = getattr(st, "st_birthtime", None)
    if bt is None and IS_WIN:
        bt = st.st_ctime  # no Windows, st_ctime é a data de criação
    return bt


def _fmt(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""


def _set_creation_time(path: str, ts: float) -> bool:
    """Ajusta a data de CRIAÇÃO da cópia (Windows). shutil.copy2 não preserva essa data."""
    if not IS_WIN or not ts:
        return False
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.SetFileTime.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                                ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    FILE_WRITE_ATTRIBUTES, OPEN_EXISTING, FLAG_BACKUP = 0x100, 3, 0x02000000
    h = k32.CreateFileW(_long(path), FILE_WRITE_ATTRIBUTES, 0x7, None, OPEN_EXISTING, FLAG_BACKUP, None)
    if h in (None, wintypes.HANDLE(-1).value):
        return False
    try:
        v = int(round(ts * 10_000_000)) + 116444736000000000
        ft = wintypes.FILETIME(v & 0xFFFFFFFF, v >> 32)
        return bool(k32.SetFileTime(h, ctypes.byref(ft), None, None))
    finally:
        k32.CloseHandle(h)


def _safe_dir(name: str) -> str:
    """Só para nomes de PASTAS criadas pelo programa (nunca para os arquivos exportados)."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .") or "imagem"


def default_order(results: list[dict]) -> list[dict]:
    """Mesma ordem padrão da aba Semelhantes: data de modificação crescente, sem data no fim."""
    return sorted(results, key=lambda r: (r["rec"].date_file is None, r["rec"].date_file or ""))


def _rel_in_folder(path: str, folder: str) -> str:
    try:
        rel = os.path.relpath(path, folder)
    except ValueError:  # outra unidade (não deve ocorrer: tudo vem da pasta analisada)
        rel = None
    if not rel or rel.startswith(".."):
        drive, tail = os.path.splitdrive(os.path.abspath(path))
        rel = os.path.join("_fora_da_pasta", drive.replace(":", ""), tail.lstrip("\\/"))
    return rel


# ---------------------------------------------------------------------------- cópia de um arquivo
def copy_preserving(src: str, dst: str, expected_sha: str | None) -> dict:
    """Copia src -> dst preservando nome, conteúdo e datas; nunca sobrescreve. Retorna o registro."""
    info = {"status": "", "sha_orig": "", "md5_orig": "", "sha_copy": "", "md5_copy": "",
            "ctime_o": "", "mtime_o": "", "atime_o": "", "ctime_c": "", "mtime_c": "",
            "dates_ok": "", "size": ""}
    st = os.stat(_long(src))
    bt = _birthtime(st)
    info.update(size=st.st_size, ctime_o=_fmt(bt), mtime_o=_fmt(st.st_mtime), atime_o=_fmt(st.st_atime))
    info["sha_orig"], info["md5_orig"] = _hashes(src)
    notes = []
    if expected_sha and info["sha_orig"] != expected_sha:
        notes.append("ORIGINAL MUDOU desde a indexação (hash atual ≠ índice)")
    if os.path.lexists(_long(dst)):
        sha_d, md5_d = _hashes(dst)
        info["sha_copy"], info["md5_copy"] = sha_d, md5_d
        if sha_d == info["sha_orig"]:
            info["status"] = "JÁ EXISTIA no destino (conteúdo idêntico) — não sobrescrito"
        else:
            info["status"] = "CONFLITO: já existe arquivo DIFERENTE no destino — não copiado"
        if notes:
            info["status"] += "; " + "; ".join(notes)
        return info
    os.makedirs(_long(os.path.dirname(dst)), exist_ok=True)
    shutil.copy2(_long(src), _long(dst))           # conteúdo + datas de modificação/acesso + atributos
    created = _set_creation_time(dst, bt)            # data de criação (Windows)
    info["sha_copy"], info["md5_copy"] = _hashes(dst)
    stc = os.stat(_long(dst))
    info["ctime_c"], info["mtime_c"] = _fmt(_birthtime(stc) if IS_WIN else None), _fmt(stc.st_mtime)
    m_ok = abs(stc.st_mtime - st.st_mtime) <= 2
    c_ok = (not IS_WIN) or (created and bt is not None and abs((_birthtime(stc) or 0) - bt) <= 2)
    info["dates_ok"] = "sim" if (m_ok and c_ok) else ("modificação sim; criação NÃO" if m_ok else "NÃO")
    if info["sha_copy"] != info["sha_orig"]:
        notes.insert(0, "FALHA: hash da cópia ≠ original")
    info["status"] = "OK" if not notes else "; ".join(notes)
    return info


# ---------------------------------------------------------------------------- exportação
def export_sets(sets: list[tuple], dest: str, folder: str, progress=None, cancelled=lambda: False) -> dict:
    """sets: [(rec_referência, [resultados])]. Retorna resumo."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(dest, f"exportacao_{stamp}")
    k = 2
    while os.path.exists(_long(root)):  # nunca reutiliza uma pasta de exportação existente
        root = os.path.join(dest, f"exportacao_{stamp}_{k}")
        k += 1
    os.makedirs(_long(root), exist_ok=False)
    total = sum(1 + len(r) for _, r in sets)
    done = 0
    summary = {"dirs": [], "files": 0, "bad": 0, "dest": root, "manifests": []}
    for n_set, (ref, results) in enumerate(sets, 1):
        if cancelled():
            break
        # pasta do conjunto (criada pelo programa); dentro dela, a estrutura ORIGINAL da pasta analisada
        sub = os.path.join(root, f"{n_set:03d}_referencia_{_safe_dir(ref.name).replace('.', '_')}")
        os.makedirs(_long(sub), exist_ok=True)
        summary["dirs"].append(sub)
        items = [(ref, "Imagem de referência", 100.0, "")] + [
            (r["rec"], r["cat"], r["score"], r["detail"]) for r in default_order(results)]
        rows, sums256, sums_md5 = [], [], []
        for i, (rec, cat, score, detail) in enumerate(items):
            if cancelled():
                break
            rel = _rel_in_folder(rec.path, folder)
            dst = os.path.join(sub, rel)
            try:
                inf = copy_preserving(rec.path, dst, rec.sha256)
            except OSError as e:
                inf = {"status": f"ERRO: {e}", "sha_orig": "", "md5_orig": "", "sha_copy": "", "md5_copy": "",
                       "ctime_o": "", "mtime_o": "", "atime_o": "", "ctime_c": "", "mtime_c": "",
                       "dates_ok": "", "size": rec.size}
            ok = inf["status"].startswith("OK") or inf["status"].startswith("JÁ EXISTIA")
            if ok:
                summary["files"] += 1
            if not inf["status"].startswith("OK"):
                summary["bad"] += 1
            if inf["sha_copy"]:
                sums256.append(f"{inf['sha_copy']} *{rel}")
                sums_md5.append(f"{inf['md5_copy']} *{rel}")
            rows.append([i, cat, f"{score:.0f}", detail, os.path.basename(rec.path), rec.path, rel,
                         inf["size"], inf["ctime_o"], inf["mtime_o"], inf["atime_o"], rec.date_taken or "",
                         rec.width, rec.height, rec.fmt, rec.sha256 or "", inf["sha_orig"], inf["sha_copy"],
                         inf["md5_orig"], inf["md5_copy"], inf["ctime_c"], inf["mtime_c"], inf["dates_ok"],
                         inf["status"]])
            done += 1
            if progress:
                progress(done, total, f"Copiando {rec.name}")
        man = os.path.join(sub, "manifesto.csv")
        with open(_long(man), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([f"{APP_NAME} {__version__}", "Exportado em", datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        "Computador", os.environ.get("COMPUTERNAME", ""), "Usuário",
                        os.environ.get("USERNAME", os.environ.get("USER", ""))])
            w.writerow(["Pasta analisada", folder])
            w.writerow(["Pasta do conjunto", sub])
            w.writerow(["Referência", ref.path, "SHA-256", ref.sha256 or ""])
            w.writerow(["Observação", "Arquivos copiados com NOME ORIGINAL e caminho relativo à pasta analisada "
                        "preservados; nenhum arquivo foi renomeado ou sobrescrito. Os originais foram apenas lidos."])
            w.writerow([])
            w.writerow(["Ordem", "Relação com a referência", "Confiança (%)", "Detalhe", "Nome do arquivo",
                        "Caminho original", "Caminho relativo (preservado na cópia)", "Tamanho (bytes)",
                        "Criação (original)", "Modificação (original)", "Último acesso (original)",
                        "Data de captura (EXIF/XMP)", "Largura", "Altura", "Formato",
                        "SHA-256 (índice)", "SHA-256 (original, no momento da exportação)", "SHA-256 (cópia)",
                        "MD5 (original)", "MD5 (cópia)", "Criação (cópia)", "Modificação (cópia)",
                        "Datas preservadas", "Conferência"])
            w.writerows(rows)
        for name, lines in (("hashes_sha256.txt", sums256), ("hashes_md5.txt", sums_md5)):
            with open(_long(os.path.join(sub, name)), "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        sha_man, _ = _hashes(man)
        summary["manifests"].append((man, sha_man))
    return summary
