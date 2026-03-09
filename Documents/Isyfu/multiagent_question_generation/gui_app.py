#!/usr/bin/env python3
"""
GUI completa para el sistema multiagente de generacion de preguntas.

Pestanas:
1. Agente Z (Rewriter) - Procesar PDFs a chunks
2. Generar Preguntas (BCDE pipeline)
3. Base de Datos - Ver, borrar preguntas
4. Exportar PDF / Excel
5. Supabase - Conexion remota, upload, topic_types

Uso:
    python gui_app.py
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

from dotenv import load_dotenv, set_key

PROJECT_ROOT = Path(__file__).resolve().parent
REWRITTEN_DIR = PROJECT_ROOT / "input_docs" / "rewritten"
DB_PATH = PROJECT_ROOT / "database" / "questions.db"
OUTPUT_DIR = PROJECT_ROOT / "output"
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


# ============================================================
# Helpers
# ============================================================

def discover_topics() -> list[dict]:
    if not REWRITTEN_DIR.exists():
        return []
    topic_re = re.compile(r"Tema-(\d+)")
    topics: dict[int, dict] = {}
    for json_file in sorted(REWRITTEN_DIR.glob("*.json")):
        if json_file.name.endswith(("_chunks_metadata.json", ".coord.json")):
            continue
        match = topic_re.search(json_file.stem)
        if not match:
            continue
        topic_num = int(match.group(1))
        if topic_num in topics:
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            total_chunks = len(data.get("chunks", []))
        except Exception:
            total_chunks = 0
        topics[topic_num] = {
            "num": topic_num, "file": json_file,
            "chunks": total_chunks, "name": json_file.stem,
        }
    return [topics[k] for k in sorted(topics)]


def discover_input_pdfs() -> list[Path]:
    input_dir = PROJECT_ROOT / "input_docs"
    if not input_dir.exists():
        return []
    pdfs = sorted(input_dir.glob("*.pdf"))
    return pdfs


def get_db_stats() -> dict:
    if not DB_PATH.exists():
        return {"total": 0, "topics": {}, "manual_review": 0}
    try:
        conn = sqlite3.connect(DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        manual = conn.execute("SELECT COUNT(*) FROM questions WHERE needs_manual_review = 1").fetchone()[0]
        cursor = conn.execute("SELECT topic, COUNT(*) FROM questions GROUP BY topic ORDER BY topic")
        topics = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return {"total": total, "topics": topics, "manual_review": manual}
    except Exception:
        return {"total": 0, "topics": {}, "manual_review": 0}


def get_questions_page(offset: int, limit: int, topic_filter: int | None = None) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if topic_filter is not None:
        cursor = conn.execute(
            "SELECT * FROM questions_with_options WHERE topic = ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (topic_filter, limit, offset)
        )
    else:
        cursor = conn.execute(
            "SELECT * FROM questions_with_options ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def delete_questions_by_topic(topic: int) -> int:
    if not DB_PATH.exists():
        return 0
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.execute("DELETE FROM questions WHERE topic = ?", (topic,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def delete_all_questions() -> int:
    if not DB_PATH.exists():
        return 0
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.execute("DELETE FROM questions")
    deleted = cursor.rowcount
    conn.execute("DELETE FROM batches")
    conn.commit()
    conn.close()
    return deleted


def _supabase_headers() -> dict | None:
    """Build Supabase auth headers. Returns None if credentials missing."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    return {
        "url": url,
        "headers": {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
    }


def supabase_get(table: str, params: dict = None) -> list[dict] | None:
    """GET request to Supabase REST API. Returns None on error."""
    try:
        import httpx
    except ImportError:
        return None
    cfg = _supabase_headers()
    if not cfg:
        return None
    try:
        resp = httpx.get(f"{cfg['url']}/rest/v1/{table}", headers=cfg["headers"], params=params or {}, timeout=15)
        if resp.status_code >= 400:
            return None
        return resp.json()
    except Exception:
        return None


def supabase_post(table: str, data: dict | list) -> list[dict] | None:
    """POST request to Supabase REST API. Returns inserted rows or None on error."""
    try:
        import httpx
    except ImportError:
        return None
    cfg = _supabase_headers()
    if not cfg:
        return None
    try:
        resp = httpx.post(f"{cfg['url']}/rest/v1/{table}", headers=cfg["headers"], json=data, timeout=60)
        if resp.status_code >= 400:
            return None
        return resp.json()
    except Exception:
        return None


def supabase_patch(table: str, params: dict, data: dict) -> list[dict] | None:
    """PATCH request to Supabase REST API. params filter the rows to update.
    Returns (rows, error_msg). error_msg is None on success."""
    try:
        import httpx
    except ImportError:
        return None, "httpx not installed"
    cfg = _supabase_headers()
    if not cfg:
        return None, "Supabase credentials missing"
    try:
        resp = httpx.patch(
            f"{cfg['url']}/rest/v1/{table}", headers=cfg["headers"],
            params=params, json=data, timeout=30,
        )
        if resp.status_code >= 400:
            return None, f"HTTP {resp.status_code}: {resp.text}"
        return resp.json(), None
    except Exception as e:
        return None, str(e)


def supabase_rpc(fn_name: str, params: dict) -> dict | None:
    """Call a Supabase RPC function. Returns the result or None on error."""
    try:
        import httpx
    except ImportError:
        return None
    cfg = _supabase_headers()
    if not cfg:
        return None
    try:
        resp = httpx.post(
            f"{cfg['url']}/rest/v1/rpc/{fn_name}",
            headers=cfg["headers"], json=params, timeout=120,
        )
        if resp.status_code >= 400:
            return {"error": resp.text, "status": resp.status_code}
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Main Application
# ============================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Generador de Preguntas - Multiagente")
        self.geometry("1150x820")
        self.minsize(1000, 700)

        self._process: subprocess.Popen | None = None
        self._running = False

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Run.TButton", font=("Helvetica", 11, "bold"))
        style.configure("Danger.TButton", foreground="red")

        self._build_ui()
        self._refresh_topics()
        self._refresh_db_stats()
        self._refresh_export_topics()

        # Auto-load Supabase topic types if credentials exist
        if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            self.after(500, self._load_topic_types)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # =========================================================
    # UI
    # =========================================================

    def _build_ui(self):
        # Main paned: notebook on top, log on bottom
        paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.notebook = ttk.Notebook(paned)
        paned.add(self.notebook, weight=3)

        self._build_rewriter_tab()
        self._build_generate_tab()
        self._build_db_tab()
        self._build_export_tab()
        self._build_supabase_tab()
        self._build_config_tab()

        # Log
        log_frame = ttk.LabelFrame(paned, text="Log")
        paned.add(log_frame, weight=2)

        self.log_text = tk.Text(
            log_frame, bg="#1e1e1e", fg="#d4d4d4",
            font=("Menlo", 10), wrap=tk.WORD, state=tk.DISABLED
        )
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom bar
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, padx=6, pady=(0, 4))
        self.btn_stop = ttk.Button(bar, text="Detener proceso", command=self._on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, padx=4)
        ttk.Button(bar, text="Limpiar log", command=self._log_clear).pack(side=tk.RIGHT, padx=4)
        self.lbl_status = ttk.Label(bar, text="Listo")
        self.lbl_status.pack(side=tk.LEFT, padx=4)

        # Progress bar
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=200)
        self.progress.pack(side=tk.LEFT, padx=12)

    # ------- Tab 1: Agente Z -------

    def _build_rewriter_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Agente Z (Rewriter) ")

        info = ttk.LabelFrame(tab, text="Procesar PDFs a chunks")
        info.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(info, text="Ejecuta el Agente Z para convertir PDFs en chunks JSON coherentes.", wraplength=700).pack(
            anchor=tk.W, padx=12, pady=4
        )

        # PDF list
        pdf_frame = ttk.LabelFrame(tab, text="PDFs en input_docs/")
        pdf_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.pdf_tree = ttk.Treeview(pdf_frame, columns=("archivo", "procesado"), show="headings", selectmode="extended")
        self.pdf_tree.heading("archivo", text="Archivo PDF")
        self.pdf_tree.heading("procesado", text="Chunks JSON")
        self.pdf_tree.column("archivo", width=400)
        self.pdf_tree.column("procesado", width=120, anchor=tk.CENTER)
        scroll = ttk.Scrollbar(pdf_frame, orient=tk.VERTICAL, command=self.pdf_tree.yview)
        self.pdf_tree.configure(yscrollcommand=scroll.set)
        self.pdf_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._refresh_pdfs()

        # Config
        cfg = ttk.Frame(tab)
        cfg.pack(fill=tk.X, padx=8, pady=4)

        self.var_z_auto_topic = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg, text="Auto-detectar tema del nombre", variable=self.var_z_auto_topic).pack(side=tk.LEFT, padx=8)
        self.var_z_force = tk.BooleanVar(value=False)
        ttk.Checkbutton(cfg, text="Forzar (reprocesar)", variable=self.var_z_force).pack(side=tk.LEFT, padx=8)
        ttk.Label(cfg, text="Limite chunks:").pack(side=tk.LEFT, padx=(16, 4))
        self.var_z_limit_chunks = tk.IntVar(value=0)
        ttk.Spinbox(cfg, from_=0, to=9999, textvariable=self.var_z_limit_chunks, width=5).pack(side=tk.LEFT)

        # Buttons
        btn = ttk.Frame(tab)
        btn.pack(pady=8)
        ttk.Button(btn, text="Ejecutar Agente Z", style="Run.TButton", command=self._on_run_rewriter).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn, text="Anadir PDFs...", command=self._on_add_pdfs).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn, text="Extraer titulos temas", command=self._on_extract_titles).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn, text="Seleccionar todos", command=lambda: self._tree_select_all(self.pdf_tree)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn, text="Refrescar", command=self._refresh_pdfs).pack(side=tk.LEFT, padx=4)

    def _refresh_pdfs(self):
        for item in self.pdf_tree.get_children():
            self.pdf_tree.delete(item)
        pdfs = discover_input_pdfs()
        for pdf in pdfs:
            # Check if rewritten JSON exists
            cache = REWRITTEN_DIR / (pdf.stem + ".json")
            status = "Si" if cache.exists() else "No"
            self.pdf_tree.insert("", tk.END, values=(pdf.name, status))
        self._pdfs = pdfs

    # ------- Tab 2: Generar -------

    def _build_generate_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Generar Preguntas ")

        top = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        top.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Topic list
        left = ttk.LabelFrame(top, text="Temas con chunks")
        top.add(left, weight=3)

        cols = ("tema", "chunks", "preguntas_bd")
        self.topic_tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="extended")
        self.topic_tree.heading("tema", text="Tema")
        self.topic_tree.heading("chunks", text="Chunks")
        self.topic_tree.heading("preguntas_bd", text="En BD")
        self.topic_tree.column("tema", width=80, anchor=tk.CENTER)
        self.topic_tree.column("chunks", width=70, anchor=tk.CENTER)
        self.topic_tree.column("preguntas_bd", width=70, anchor=tk.CENTER)
        scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.topic_tree.yview)
        self.topic_tree.configure(yscrollcommand=scroll.set)
        self.topic_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Config panel
        right = ttk.LabelFrame(top, text="Configuracion")
        top.add(right, weight=2)

        r = 0
        for label, var_name, default, low, high in [
            ("Max preguntas por tema (0=todas):", "var_max_questions", 0, 0, 9999),
            ("Preguntas/chunk:", "var_qpc", 1, 1, 10),
            ("Academy ID:", "var_academy", 1, 1, 100),
        ]:
            ttk.Label(right, text=label).grid(row=r, column=0, sticky=tk.W, padx=8, pady=3)
            var = tk.IntVar(value=default)
            setattr(self, var_name, var)
            ttk.Spinbox(right, from_=low, to=high, textvariable=var, width=6).grid(row=r, column=1, sticky=tk.W, padx=8, pady=3)
            r += 1

        for label, var_name, default in [
            ("Saltar Agente C", "var_skip_c", False),
            ("Forzar (ignorar cache)", "var_force", False),
            ("PDF unico combinado", "var_single_pdf", False),
        ]:
            var = tk.BooleanVar(value=default)
            setattr(self, var_name, var)
            ttk.Checkbutton(right, text=label, variable=var).grid(row=r, column=0, columnspan=2, sticky=tk.W, padx=8, pady=2)
            r += 1

        r += 1
        bf = ttk.Frame(right)
        bf.grid(row=r, column=0, columnspan=2, pady=10, padx=8)
        ttk.Button(bf, text="Generar Preguntas", style="Run.TButton", command=self._on_generate).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text="DETENER", style="Danger.TButton", command=self._on_stop).pack(side=tk.LEFT, padx=4)

        r += 1
        bf2 = ttk.Frame(right)
        bf2.grid(row=r, column=0, columnspan=2, padx=8)
        ttk.Button(bf2, text="Sel. todos", command=lambda: self._tree_select_all(self.topic_tree)).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf2, text="Deseleccionar", command=lambda: self.topic_tree.selection_remove(*self.topic_tree.get_children())).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf2, text="Refrescar", command=self._refresh_topics).pack(side=tk.LEFT, padx=2)

    # ------- Tab 3: BD -------

    def _build_db_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Base de Datos ")

        # Stats bar
        stats = ttk.Frame(tab)
        stats.pack(fill=tk.X, padx=8, pady=6)
        self.lbl_db_total = ttk.Label(stats, text="Total: -", font=("Helvetica", 12, "bold"))
        self.lbl_db_total.pack(side=tk.LEFT, padx=8)
        self.lbl_db_review = ttk.Label(stats, text="Revision manual: -")
        self.lbl_db_review.pack(side=tk.LEFT, padx=8)
        ttk.Button(stats, text="Refrescar", command=self._refresh_db_stats).pack(side=tk.RIGHT, padx=4)

        # Topic filter + delete buttons
        action_bar = ttk.Frame(tab)
        action_bar.pack(fill=tk.X, padx=8, pady=2)

        ttk.Label(action_bar, text="Filtrar tema:").pack(side=tk.LEFT, padx=4)
        self.var_db_filter = tk.StringVar(value="Todos")
        self.combo_db_filter = ttk.Combobox(action_bar, textvariable=self.var_db_filter, width=10, state="readonly")
        self.combo_db_filter.pack(side=tk.LEFT, padx=4)
        self.combo_db_filter.bind("<<ComboboxSelected>>", lambda _: self._refresh_questions_view())

        ttk.Separator(action_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)

        ttk.Button(action_bar, text="Formatear referencias", command=self._on_format_references).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_bar, text="Sync referencias a remoto", command=self._on_sync_references_remote).pack(side=tk.LEFT, padx=4)
        ttk.Separator(action_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        ttk.Button(action_bar, text="Borrar tema seleccionado", style="Danger.TButton", command=self._on_delete_topic).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_bar, text="Borrar TODAS las preguntas", style="Danger.TButton", command=self._on_delete_all).pack(side=tk.LEFT, padx=4)

        # Topic summary tree
        top_paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        top_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: topics summary
        summary_frame = ttk.LabelFrame(top_paned, text="Preguntas por tema")
        top_paned.add(summary_frame, weight=1)

        self.db_topic_tree = ttk.Treeview(summary_frame, columns=("tema", "cantidad"), show="headings", selectmode="browse")
        self.db_topic_tree.heading("tema", text="Tema")
        self.db_topic_tree.heading("cantidad", text="Preguntas")
        self.db_topic_tree.column("tema", width=80, anchor=tk.CENTER)
        self.db_topic_tree.column("cantidad", width=80, anchor=tk.CENTER)
        self.db_topic_tree.pack(fill=tk.BOTH, expand=True)

        # Right: questions detail
        detail_frame = ttk.LabelFrame(top_paned, text="Preguntas")
        top_paned.add(detail_frame, weight=3)

        q_cols = ("id", "pregunta", "correcta", "tema", "review")
        self.q_tree = ttk.Treeview(detail_frame, columns=q_cols, show="headings", selectmode="browse")
        self.q_tree.heading("id", text="ID")
        self.q_tree.heading("pregunta", text="Pregunta")
        self.q_tree.heading("correcta", text="Correcta")
        self.q_tree.heading("tema", text="Tema")
        self.q_tree.heading("review", text="Review")
        self.q_tree.column("id", width=45, anchor=tk.CENTER)
        self.q_tree.column("pregunta", width=400)
        self.q_tree.column("correcta", width=50, anchor=tk.CENTER)
        self.q_tree.column("tema", width=55, anchor=tk.CENTER)
        self.q_tree.column("review", width=55, anchor=tk.CENTER)
        q_scroll = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.q_tree.yview)
        self.q_tree.configure(yscrollcommand=q_scroll.set)
        self.q_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        q_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Pagination
        pg = ttk.Frame(tab)
        pg.pack(fill=tk.X, padx=8, pady=4)
        self._q_offset = 0
        self._q_page_size = 50
        ttk.Button(pg, text="<< Anterior", command=self._q_prev_page).pack(side=tk.LEFT, padx=4)
        self.lbl_page = ttk.Label(pg, text="Pagina 1")
        self.lbl_page.pack(side=tk.LEFT, padx=8)
        ttk.Button(pg, text="Siguiente >>", command=self._q_next_page).pack(side=tk.LEFT, padx=4)

    # ------- Tab 4: Exportar -------

    def _build_export_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Exportar PDF/Excel ")

        top = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        top.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: topic selection for export
        left = ttk.LabelFrame(top, text="Temas a exportar")
        top.add(left, weight=2)

        self.export_topic_tree = ttk.Treeview(
            left, columns=("tema", "preguntas"), show="headings", selectmode="extended"
        )
        self.export_topic_tree.heading("tema", text="Tema")
        self.export_topic_tree.heading("preguntas", text="Preguntas")
        self.export_topic_tree.column("tema", width=100, anchor=tk.CENTER)
        self.export_topic_tree.column("preguntas", width=80, anchor=tk.CENTER)
        et_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.export_topic_tree.yview)
        self.export_topic_tree.configure(yscrollcommand=et_scroll.set)
        self.export_topic_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        et_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        et_btns = ttk.Frame(left)
        et_btns.pack(fill=tk.X, pady=4)
        ttk.Button(et_btns, text="Sel. todos", command=lambda: self._tree_select_all(self.export_topic_tree)).pack(side=tk.LEFT, padx=4)
        ttk.Button(et_btns, text="Refrescar", command=self._refresh_export_topics).pack(side=tk.LEFT, padx=4)

        # Right: config
        right = ttk.Frame(top)
        top.add(right, weight=3)

        fmt = ttk.LabelFrame(right, text="Formato de salida")
        fmt.pack(fill=tk.X, padx=4, pady=4)
        self.var_export_fmt = tk.StringVar(value="excel")
        for val, txt in [("excel", "Excel (.xlsx)"), ("pdf", "PDF"), ("both", "Ambos")]:
            ttk.Radiobutton(fmt, text=txt, variable=self.var_export_fmt, value=val).pack(side=tk.LEFT, padx=12, pady=4)

        ef = ttk.LabelFrame(right, text="Formato Excel")
        ef.pack(fill=tk.X, padx=4, pady=4)
        self.var_excel_fmt = tk.StringVar(value="with_solutions")
        for val, txt in [("exam", "Solo examen"), ("with_solutions", "Con soluciones"), ("study_guide", "Guia completa")]:
            ttk.Radiobutton(ef, text=txt, variable=self.var_excel_fmt, value=val).pack(anchor=tk.W, padx=12, pady=2)

        pf = ttk.LabelFrame(right, text="Formato PDF")
        pf.pack(fill=tk.X, padx=4, pady=4)
        self.var_pdf_fmt = tk.StringVar(value="study_guide")
        for val, txt in [("exam", "Solo examen"), ("with_solutions", "Con soluciones"),
                         ("study_guide", "Guia estudio"), ("study_guide_with_chunks", "Guia + chunks")]:
            ttk.Radiobutton(pf, text=txt, variable=self.var_pdf_fmt, value=val).pack(anchor=tk.W, padx=12, pady=2)

        # Output directory
        dir_frame = ttk.LabelFrame(right, text="Directorio de salida")
        dir_frame.pack(fill=tk.X, padx=4, pady=4)
        self.var_export_dir = tk.StringVar(value=str(OUTPUT_DIR))
        dir_row = ttk.Frame(dir_frame)
        dir_row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Entry(dir_row, textvariable=self.var_export_dir, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(dir_row, text="Elegir...", command=self._choose_export_dir).pack(side=tk.LEFT)

        self.btn_export = ttk.Button(right, text="Exportar", style="Run.TButton", command=self._on_export)
        self.btn_export.pack(pady=12)

    def _refresh_export_topics(self):
        for item in self.export_topic_tree.get_children():
            self.export_topic_tree.delete(item)
        stats = get_db_stats()
        for topic, count in sorted(stats["topics"].items()):
            self.export_topic_tree.insert("", tk.END, values=(f"Tema {topic}", count), iid=str(topic))

    def _choose_export_dir(self):
        d = filedialog.askdirectory(title="Elegir directorio de salida", initialdir=self.var_export_dir.get())
        if d:
            self.var_export_dir.set(d)

    # ------- Tab 5: Supabase -------

    def _build_supabase_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Supabase ")

        # Connection
        conn_frame = ttk.LabelFrame(tab, text="Conexion remota")
        conn_frame.pack(fill=tk.X, padx=8, pady=8)

        self.var_supa_url = tk.StringVar(value=os.getenv("SUPABASE_URL", ""))
        self.var_supa_key = tk.StringVar(value=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))

        r = 0
        ttk.Label(conn_frame, text="SUPABASE_URL:").grid(row=r, column=0, sticky=tk.W, padx=8, pady=4)
        ttk.Entry(conn_frame, textvariable=self.var_supa_url, width=60).grid(row=r, column=1, sticky=tk.EW, padx=8, pady=4)
        r += 1
        ttk.Label(conn_frame, text="SERVICE_ROLE_KEY:").grid(row=r, column=0, sticky=tk.W, padx=8, pady=4)
        key_entry = ttk.Entry(conn_frame, textvariable=self.var_supa_key, width=60, show="*")
        key_entry.grid(row=r, column=1, sticky=tk.EW, padx=8, pady=4)
        self._supa_key_visible = False
        def toggle_key():
            self._supa_key_visible = not self._supa_key_visible
            key_entry.configure(show="" if self._supa_key_visible else "*")
        ttk.Button(conn_frame, text="Mostrar/Ocultar", command=toggle_key).grid(row=r, column=2, padx=4)

        r += 1
        btn_conn = ttk.Frame(conn_frame)
        btn_conn.grid(row=r, column=0, columnspan=3, pady=8)
        ttk.Button(btn_conn, text="Guardar en .env", command=self._save_supabase_env).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_conn, text="Probar conexion", command=self._test_supabase).pack(side=tk.LEFT, padx=4)

        conn_frame.columnconfigure(1, weight=1)

        # Main content: left = remote browser, right = upload from local
        supa_paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        supa_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # ---- Left: Remote browser ----
        left = ttk.Frame(supa_paned)
        supa_paned.add(left, weight=3)

        tt_frame = ttk.LabelFrame(left, text="Topic Types (Study)")
        tt_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self.tt_tree = ttk.Treeview(tt_frame, columns=("id", "nombre", "academy", "desc"), show="headings", height=5)
        self.tt_tree.heading("id", text="ID")
        self.tt_tree.heading("nombre", text="Nombre")
        self.tt_tree.heading("academy", text="Academy")
        self.tt_tree.heading("desc", text="Descripcion")
        self.tt_tree.column("id", width=40, anchor=tk.CENTER)
        self.tt_tree.column("nombre", width=200)
        self.tt_tree.column("academy", width=60, anchor=tk.CENTER)
        self.tt_tree.column("desc", width=200)
        tt_scroll = ttk.Scrollbar(tt_frame, orient=tk.VERTICAL, command=self.tt_tree.yview)
        self.tt_tree.configure(yscrollcommand=tt_scroll.set)
        self.tt_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tt_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        topics_frame = ttk.LabelFrame(left, text="Topics del tipo seleccionado")
        topics_frame.pack(fill=tk.BOTH, expand=True)

        self.supa_topics_tree = ttk.Treeview(
            topics_frame, columns=("id", "order", "nombre", "preguntas"), show="headings", height=6
        )
        self.supa_topics_tree.heading("id", text="ID")
        self.supa_topics_tree.heading("order", text="Orden")
        self.supa_topics_tree.heading("nombre", text="Nombre")
        self.supa_topics_tree.heading("preguntas", text="Preguntas")
        self.supa_topics_tree.column("id", width=40, anchor=tk.CENTER)
        self.supa_topics_tree.column("order", width=50, anchor=tk.CENTER)
        self.supa_topics_tree.column("nombre", width=250)
        self.supa_topics_tree.column("preguntas", width=70, anchor=tk.CENTER)
        st_scroll = ttk.Scrollbar(topics_frame, orient=tk.VERTICAL, command=self.supa_topics_tree.yview)
        self.supa_topics_tree.configure(yscrollcommand=st_scroll.set)
        self.supa_topics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        st_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tt_tree.bind("<<TreeviewSelect>>", self._on_tt_select)

        btn_left = ttk.Frame(left)
        btn_left.pack(fill=tk.X, pady=4)
        ttk.Button(btn_left, text="Cargar Topic Types", command=self._load_topic_types).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_left, text="Sincronizar temas con Supabase", command=self._on_create_topics_supabase).pack(side=tk.LEFT, padx=4)

        # ---- Right: Upload from local DB ----
        right = ttk.LabelFrame(supa_paned, text="Subir preguntas a Supabase")
        supa_paned.add(right, weight=2)

        ttk.Label(right, text="Selecciona temas locales para subir:", font=("Helvetica", 10)).pack(anchor=tk.W, padx=8, pady=(8, 4))

        upload_tree_frame = ttk.Frame(right)
        upload_tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.upload_topic_tree = ttk.Treeview(
            upload_tree_frame, columns=("tema", "local", "remoto"), show="headings", selectmode="extended", height=8
        )
        self.upload_topic_tree.heading("tema", text="Tema")
        self.upload_topic_tree.heading("local", text="Local (BD)")
        self.upload_topic_tree.heading("remoto", text="Remoto")
        self.upload_topic_tree.column("tema", width=80, anchor=tk.CENTER)
        self.upload_topic_tree.column("local", width=70, anchor=tk.CENTER)
        self.upload_topic_tree.column("remoto", width=70, anchor=tk.CENTER)
        ut_scroll = ttk.Scrollbar(upload_tree_frame, orient=tk.VERTICAL, command=self.upload_topic_tree.yview)
        self.upload_topic_tree.configure(yscrollcommand=ut_scroll.set)
        self.upload_topic_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ut_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        upload_btns = ttk.Frame(right)
        upload_btns.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(upload_btns, text="Sel. todos", command=lambda: self._tree_select_all(self.upload_topic_tree)).pack(side=tk.LEFT, padx=4)
        ttk.Button(upload_btns, text="Refrescar", command=self._refresh_upload_topics).pack(side=tk.LEFT, padx=4)

        ttk.Button(right, text="Subir preguntas seleccionadas", style="Run.TButton",
                   command=self._on_upload_selected).pack(pady=8)

    # =========================================================
    # Helpers
    # =========================================================

    def _tree_select_all(self, tree: ttk.Treeview):
        for item in tree.get_children():
            tree.selection_add(item)

    def _log(self, text: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _log_clear(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _log_ts(self, text: str):
        self.after(0, self._log, text)

    def _set_running(self, running: bool):
        self._running = running
        self.btn_stop.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.lbl_status.configure(text="Ejecutando..." if running else "Listo")
        if running:
            self.progress.start(15)
        else:
            self.progress.stop()

    def _set_running_ts(self, running: bool):
        self.after(0, self._set_running, running)

    def _run_subprocess(self, cmd: list[str], on_done=None, stdin_text: str = None):
        """Run a subprocess in a thread, streaming output to log."""
        def _worker():
            try:
                self._log_ts(f"$ {' '.join(cmd)}\n\n")
                self._process = subprocess.Popen(
                    cmd, cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE if stdin_text else None,
                    text=True, bufsize=1,
                    preexec_fn=os.setsid,
                )
                if stdin_text:
                    self._process.stdin.write(stdin_text)
                    self._process.stdin.close()
                for line in iter(self._process.stdout.readline, ""):
                    if not self._running:
                        self._process.terminate()
                        break
                    self._log_ts(line)
                self._process.wait()
                rc = self._process.returncode
                if rc != 0:
                    self._log_ts(f"\n[Exit code: {rc}]\n")
                self._process = None
            except Exception as e:
                self._log_ts(f"\nError: {e}\n")
            finally:
                self._set_running_ts(False)
                if on_done:
                    self.after(300, on_done)

        self._log_clear()
        self._set_running(True)
        threading.Thread(target=_worker, daemon=True).start()

    def _on_stop(self):
        self._running = False
        if self._process:
            try:
                import signal
                # Kill the entire process group to stop child processes too
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    self._process.kill()
                except Exception:
                    pass
        self._log_ts("\n--- Detenido ---\n")
        self._set_running(False)

    # =========================================================
    # Refresh data
    # =========================================================

    def _refresh_topics(self):
        for item in self.topic_tree.get_children():
            self.topic_tree.delete(item)
        self._topics = discover_topics()
        db_stats = get_db_stats()
        for t in self._topics:
            in_db = db_stats["topics"].get(t["num"], 0)
            self.topic_tree.insert("", tk.END, values=(f"Tema {t['num']}", t["chunks"], in_db))

    def _refresh_db_stats(self):
        stats = get_db_stats()
        self.lbl_db_total.configure(text=f"Total: {stats['total']}")
        self.lbl_db_review.configure(text=f"Revision manual: {stats['manual_review']}")

        for item in self.db_topic_tree.get_children():
            self.db_topic_tree.delete(item)
        topic_list = ["Todos"] + [str(t) for t in sorted(stats["topics"].keys())]
        self.combo_db_filter.configure(values=topic_list)

        for topic, count in sorted(stats["topics"].items()):
            self.db_topic_tree.insert("", tk.END, values=(f"Tema {topic}", count))

        self._q_offset = 0
        self._refresh_questions_view()

    def _refresh_questions_view(self):
        for item in self.q_tree.get_children():
            self.q_tree.delete(item)
        filt = self.var_db_filter.get()
        topic_filter = int(filt) if filt != "Todos" else None
        rows = get_questions_page(self._q_offset, self._q_page_size, topic_filter)
        for r in rows:
            solution = r.get("solution", "?")
            letter = chr(64 + solution) if isinstance(solution, int) and 1 <= solution <= 4 else "?"
            review = "Si" if r.get("needs_manual_review") else ""
            q_text = (r.get("question") or "")[:120]
            self.q_tree.insert("", tk.END, values=(r.get("id"), q_text, letter, r.get("topic"), review))
        page = (self._q_offset // self._q_page_size) + 1
        self.lbl_page.configure(text=f"Pagina {page} ({len(rows)} mostradas)")

    def _q_next_page(self):
        self._q_offset += self._q_page_size
        self._refresh_questions_view()

    def _q_prev_page(self):
        self._q_offset = max(0, self._q_offset - self._q_page_size)
        self._refresh_questions_view()

    # =========================================================
    # Actions: Agente Z
    # =========================================================

    def _on_extract_titles(self):
        """Run extract_topic_titles.py using OpenAI Vision on selected PDFs or all."""
        if self._running:
            messagebox.showwarning("Ocupado", "Espera a que termine el proceso actual.")
            return

        # Check selected PDFs — extract tema numbers from them
        selected = self.pdf_tree.selection()
        topic_nums = []
        if selected:
            all_children = self.pdf_tree.get_children()
            for item in selected:
                idx = all_children.index(item)
                pdf_path = self._pdfs[idx]
                parts = pdf_path.stem.split("-")
                try:
                    topic_nums.append(int(parts[1]))
                except (IndexError, ValueError):
                    pass

        if topic_nums:
            names = ", ".join(str(n) for n in sorted(topic_nums))
            msg = f"Extraer titulos de temas seleccionados ({names}) usando OpenAI Vision?"
        else:
            msg = "Extraer titulos de TODOS los temas usando OpenAI Vision?\n\n(Selecciona PDFs para extraer solo algunos)"

        if not messagebox.askyesno("Extraer titulos", msg):
            return

        cmd = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "extract_topic_titles.py"), "--force"]
        if topic_nums:
            cmd.extend(["--topics", ",".join(str(n) for n in topic_nums)])
        self._run_subprocess(cmd, on_done=self._refresh_topics)

    def _on_add_pdfs(self):
        """Let user select PDF files to copy into input_docs/."""
        import shutil
        files = filedialog.askopenfilenames(
            title="Seleccionar PDFs para procesar",
            filetypes=[("PDF", "*.pdf"), ("Todos", "*.*")],
        )
        if not files:
            return
        input_dir = PROJECT_ROOT / "input_docs"
        input_dir.mkdir(exist_ok=True)
        added = 0
        for f in files:
            src = Path(f)
            dst = input_dir / src.name
            if dst.exists():
                if not messagebox.askyesno("Archivo existente", f"{src.name} ya existe. Sobreescribir?"):
                    continue
            shutil.copy2(str(src), str(dst))
            added += 1
        if added:
            self._log(f"Anadidos {added} PDF(s) a input_docs/\n")
            self._refresh_pdfs()

    def _on_run_rewriter(self):
        if self._running:
            messagebox.showwarning("Ocupado", "Espera a que termine el proceso actual.")
            return
        selected = self.pdf_tree.selection()
        if not selected:
            messagebox.showwarning("Sin seleccion", "Selecciona al menos un PDF.")
            return

        all_children = self.pdf_tree.get_children()
        indices = [all_children.index(item) for item in selected]
        pdf_names = [self._pdfs[i].name for i in indices]

        if not messagebox.askyesno("Agente Z", f"Procesar {len(pdf_names)} PDF(s) con Agente Z?"):
            return

        # Run sequentially per PDF
        def _run():
            try:
                for i, pdf_name in enumerate(pdf_names):
                    if not self._running:
                        break
                    self._log_ts(f"\n{'='*60}\n  [{i+1}/{len(pdf_names)}] {pdf_name}\n{'='*60}\n")

                    cmd = [
                        sys.executable, str(PROJECT_ROOT / "run_rewriter.py"),
                        "--auto-topic" if self.var_z_auto_topic.get() else "--topic", "1",
                    ]
                    if self.var_z_auto_topic.get():
                        cmd = [sys.executable, str(PROJECT_ROOT / "run_rewriter.py"), "--auto-topic"]
                    if self.var_z_force.get():
                        cmd.append("--force")
                    lc = self.var_z_limit_chunks.get()
                    if lc > 0:
                        cmd.extend(["--limit-chunks", str(lc)])
                    # Limit to just this file by using --limit 1 and hoping it's next,
                    # but actually run_rewriter processes all pending. We filter with name.
                    # The safest way: process all but we'll show the user it's running.
                    # Actually let's just run the whole rewriter with --limit to 1 file approach.
                    # run_rewriter doesn't have a --doc filter, so we run for all pending.

                    self._log_ts(f"$ {' '.join(cmd)}\n\n")
                    self._process = subprocess.Popen(
                        cmd, cwd=str(PROJECT_ROOT),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                        preexec_fn=os.setsid,
                    )
                    for line in iter(self._process.stdout.readline, ""):
                        if not self._running:
                            break
                        self._log_ts(line)
                    self._process.wait()
                    self._process = None
                    break  # run_rewriter handles all files, so run once
            except Exception as e:
                self._log_ts(f"\nError: {e}\n")
            finally:
                self._set_running_ts(False)
                self.after(300, self._refresh_pdfs)
                self.after(500, self._refresh_topics)

        self._log_clear()
        self._set_running(True)
        threading.Thread(target=_run, daemon=True).start()

    # =========================================================
    # Actions: Generate
    # =========================================================

    def _get_selected_topics(self) -> list[dict]:
        selected = self.topic_tree.selection()
        if not selected:
            return []
        all_children = self.topic_tree.get_children()
        indices = [all_children.index(item) for item in selected]
        return [self._topics[i] for i in indices]

    def _on_generate(self):
        if self._running:
            messagebox.showwarning("Ocupado", "Espera a que termine el proceso actual.")
            return
        selected = self._get_selected_topics()
        if not selected:
            messagebox.showwarning("Sin seleccion", "Selecciona al menos un tema.")
            return

        names = ", ".join(f"Tema {t['num']}" for t in selected)
        if not messagebox.askyesno("Generar", f"Generar preguntas para {len(selected)} tema(s)?\n{names}"):
            return

        def _run():
            try:
                for i, t in enumerate(selected):
                    if not self._running:
                        break
                    num = t["num"]
                    self._log_ts(f"\n{'='*60}\n  TEMA {num} ({i+1}/{len(selected)})\n{'='*60}\n")
                    cmd = [
                        sys.executable, str(PROJECT_ROOT / "run_bcde_pipeline.py"),
                        "--topic", str(num), "--academy", str(self.var_academy.get()),
                        "--doc", f"Tema-{num}",
                    ]
                    if self.var_skip_c.get(): cmd.append("--skip-agent-c")
                    if self.var_force.get(): cmd.append("--force")
                    if self.var_single_pdf.get(): cmd.append("--single-pdf")

                    # Calculate questions-per-chunk and limit-chunks from max questions
                    max_q = self.var_max_questions.get()
                    n_chunks = t.get("chunks", 0)
                    if max_q > 0 and n_chunks > 0:
                        import math
                        qpc = max(1, math.ceil(max_q / n_chunks))
                        limit_chunks = math.ceil(max_q / qpc)
                        cmd.extend(["--questions-per-chunk", str(qpc)])
                        cmd.extend(["--limit-chunks", str(limit_chunks)])
                        self._log_ts(f"  -> {max_q} preguntas: {qpc} preg/chunk x {limit_chunks} chunks\n")
                    else:
                        cmd.extend(["--questions-per-chunk", str(self.var_qpc.get())])

                    self._log_ts(f"$ {' '.join(cmd)}\n\n")
                    self._process = subprocess.Popen(
                        cmd, cwd=str(PROJECT_ROOT),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                        preexec_fn=os.setsid,
                    )
                    for line in iter(self._process.stdout.readline, ""):
                        if not self._running:
                            break
                        self._log_ts(line)
                    self._process.wait()
                    if self._process and self._process.returncode != 0:
                        self._log_ts(f"\n[Exit code: {self._process.returncode}]\n")
                    self._process = None
            except Exception as e:
                self._log_ts(f"\nError: {e}\n")
            finally:
                self._set_running_ts(False)
                self.after(500, self._refresh_db_stats)
                self.after(500, self._refresh_topics)

        self._log_clear()
        self._set_running(True)
        threading.Thread(target=_run, daemon=True).start()

    # =========================================================
    # Actions: DB delete
    # =========================================================

    def _on_delete_topic(self):
        sel = self.db_topic_tree.selection()
        if not sel:
            messagebox.showwarning("Sin seleccion", "Selecciona un tema de la lista izquierda.")
            return
        vals = self.db_topic_tree.item(sel[0], "values")
        topic_str = vals[0]  # "Tema X"
        topic_num = int(topic_str.replace("Tema ", ""))
        count = int(vals[1])

        if not messagebox.askyesno(
            "Borrar tema",
            f"Borrar {count} preguntas del {topic_str}?\n\nEsta accion no se puede deshacer."
        ):
            return
        deleted = delete_questions_by_topic(topic_num)
        self._log(f"Borradas {deleted} preguntas del Tema {topic_num}\n")
        self._refresh_db_stats()
        self._refresh_topics()

    def _on_delete_all(self):
        stats = get_db_stats()
        total = stats["total"]
        if total == 0:
            messagebox.showinfo("Vacio", "No hay preguntas en la base de datos.")
            return
        if not messagebox.askyesno(
            "BORRAR TODO",
            f"BORRAR las {total} preguntas de la base de datos?\n\nEsta accion NO se puede deshacer."
        ):
            return
        # Double confirmation
        if not messagebox.askyesno("Confirmar borrado total", "Estas SEGURO? Se perderan todas las preguntas."):
            return
        deleted = delete_all_questions()
        self._log(f"Borradas {deleted} preguntas (todas).\n")
        self._refresh_db_stats()
        self._refresh_topics()

    # =========================================================
    # Actions: Format references (Agent G)
    # =========================================================

    def _on_format_references(self):
        """Run Agent G to format problematic article references."""
        if self._running:
            messagebox.showwarning("Ocupado", "Espera a que termine el proceso actual.")
            return

        # Determine scope from filter
        filt = self.var_db_filter.get()
        topic = int(filt) if filt != "Todos" else None
        scope = f"Tema {topic}" if topic else "todos los temas"

        if not messagebox.askyesno(
            "Formatear referencias",
            f"Escanear y formatear las referencias de {scope}?\n\n"
            "El Agente G detectara artefactos de OCR, marcadores de pagina, "
            "texto de libros y reformateara las referencias problematicas."
        ):
            return

        def _run():
            try:
                from agents.agent_g_reference_formatter import AgentG
                agent = AgentG()

                # Scan
                self._log_ts(f"Escaneando referencias de {scope}...\n")
                problematic = agent.scan_topic(topic)
                self._log_ts(
                    f"  Revisadas: {agent.stats['total_checked']} | "
                    f"Correctas: {agent.stats['skipped_ok']} | "
                    f"Necesitan formato: {len(problematic)}\n\n"
                )

                if not problematic:
                    self._log_ts("Todas las referencias son correctas.\n")
                    return

                # Collect formatted questions for remote sync
                formatted_questions = []  # list of (question_text, new_article, topic_num)

                for i, q in enumerate(problematic, 1):
                    if not self._running:
                        self._log_ts("\nDetenido por el usuario.\n")
                        break
                    self._log_ts(
                        f"[{i}/{len(problematic)}] Q{q['id']} (Tema {q['topic']})\n"
                        f"  Razones: {', '.join(q['reasons'])}\n"
                        f"  Pregunta: {q['question'][:80]}...\n"
                    )
                    formatted = agent.format_question(q["id"], q["question"], q["article"])
                    if formatted:
                        agent.update_article(q["id"], formatted)
                        formatted_questions.append((q["question"], formatted, q["topic"]))
                        self._log_ts(f"  -> Formateado OK ({len(q['article'])} -> {len(formatted)} chars)\n\n")
                    else:
                        self._log_ts(f"  -> Error, se mantiene original\n\n")

                self._log_ts(
                    f"\nResumen local: {agent.stats['formatted']} formateadas, "
                    f"{agent.stats['errors']} errores, "
                    f"{agent.stats['skipped_ok']} correctas\n"
                )

                # Sync to Supabase: update article field for matching questions
                if formatted_questions:
                    self._log_ts(f"\nSincronizando {len(formatted_questions)} referencias con Supabase...\n")
                    updated_remote = 0
                    skipped_remote = 0

                    # Group by topic for efficiency
                    by_topic = {}
                    for q_text, new_article, t_num in formatted_questions:
                        by_topic.setdefault(t_num, []).append((q_text, new_article))

                    for t_num, items in sorted(by_topic.items()):
                        if not self._running:
                            break
                        # Find remote questions for this topic to get their IDs
                        # We need to match by question text
                        remote_qs = supabase_get("questions", {
                            "select": "id,question",
                            "limit": "10000",
                        })
                        if not remote_qs:
                            self._log_ts(f"  Tema {t_num}: no se pudo obtener preguntas remotas\n")
                            continue

                        # Build lookup: lowercase question -> remote id
                        remote_lookup = {
                            q["question"].strip().lower(): q["id"]
                            for q in remote_qs if q.get("question")
                        }

                        for q_text, new_article in items:
                            if not self._running:
                                break
                            remote_id = remote_lookup.get(q_text.strip().lower())
                            if remote_id:
                                clean_article = new_article.replace("\x00", "").strip()
                                result, err = supabase_patch(
                                    "questions",
                                    {"id": f"eq.{remote_id}"},
                                    {"article": clean_article},
                                )
                                if err is None:
                                    updated_remote += 1
                                else:
                                    self._log_ts(f"  Error remoto Q{remote_id}: {err}\n")
                            else:
                                skipped_remote += 1

                    self._log_ts(
                        f"  Remoto: {updated_remote} actualizadas, "
                        f"{skipped_remote} no encontradas en Supabase\n"
                    )

            except Exception as e:
                self._log_ts(f"\nError: {e}\n")
            finally:
                self._set_running_ts(False)
                self.after(500, self._refresh_db_stats)

        self._log_clear()
        self._set_running(True)
        threading.Thread(target=_run, daemon=True).start()

    def _on_sync_references_remote(self):
        """Sync all local article fields to Supabase (update existing questions)."""
        if self._running:
            messagebox.showwarning("Ocupado", "Espera a que termine el proceso actual.")
            return

        filt = self.var_db_filter.get()
        topic = int(filt) if filt != "Todos" else None
        scope = f"Tema {topic}" if topic else "todos los temas"

        if not messagebox.askyesno(
            "Sync referencias a remoto",
            f"Actualizar el campo 'article' en Supabase para {scope}?\n\n"
            "Esto sincroniza las referencias locales con las remotas\n"
            "(solo preguntas que ya existen en Supabase)."
        ):
            return

        def _run():
            try:
                conn = sqlite3.connect(DB_PATH)
                conn.row_factory = sqlite3.Row
                if topic:
                    rows = conn.execute(
                        "SELECT question, article FROM questions WHERE topic = ? AND article IS NOT NULL AND article != ''",
                        (topic,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT question, article FROM questions WHERE article IS NOT NULL AND article != ''"
                    ).fetchall()
                conn.close()

                self._log_ts(f"Sincronizando {len(rows)} referencias de {scope} con Supabase...\n")

                # Get all remote questions to match by text
                remote_qs = supabase_get("questions", {
                    "select": "id,question,article",
                    "limit": "10000",
                })
                if not remote_qs:
                    self._log_ts("Error: no se pudieron obtener preguntas remotas.\n")
                    return

                remote_lookup = {}
                for q in remote_qs:
                    if q.get("question"):
                        remote_lookup[q["question"].strip().lower()] = {
                            "id": q["id"],
                            "article": q.get("article") or "",
                        }

                self._log_ts(f"  {len(remote_lookup)} preguntas encontradas en Supabase\n")

                updated = 0
                skipped_same = 0
                not_found = 0

                for row in rows:
                    if not self._running:
                        self._log_ts("\nDetenido por el usuario.\n")
                        break

                    q_text = (row["question"] or "").strip()
                    local_article = (row["article"] or "").replace("\x00", "").strip()
                    remote = remote_lookup.get(q_text.lower())

                    if not remote:
                        not_found += 1
                        continue

                    # Skip if already the same
                    if remote["article"].strip() == local_article:
                        skipped_same += 1
                        continue

                    result, err = supabase_patch(
                        "questions",
                        {"id": f"eq.{remote['id']}"},
                        {"article": local_article},
                    )
                    if err is None:
                        updated += 1
                    else:
                        self._log_ts(f"  Error Q{remote['id']}: {err}\n")

                self._log_ts(
                    f"\nResultado:\n"
                    f"  Actualizadas en remoto: {updated}\n"
                    f"  Ya iguales (sin cambio): {skipped_same}\n"
                    f"  No encontradas en remoto: {not_found}\n"
                )
            except Exception as e:
                self._log_ts(f"\nError: {e}\n")
            finally:
                self._set_running_ts(False)

        self._log_clear()
        self._set_running(True)
        threading.Thread(target=_run, daemon=True).start()

    # =========================================================
    # Actions: Export
    # =========================================================

    def _on_export(self):
        if self._running:
            messagebox.showwarning("Ocupado", "Espera a que termine el proceso actual.")
            return

        # Get selected topics
        sel = self.export_topic_tree.selection()
        if not sel:
            messagebox.showwarning("Sin seleccion", "Selecciona al menos un tema de la lista.")
            return

        self._export_selected_topics = [int(s) for s in sel]
        self._log_clear()
        self._set_running(True)
        threading.Thread(target=self._run_export, daemon=True).start()

    def _run_export(self):
        try:
            fmt = self.var_export_fmt.get()
            selected_topics = self._export_selected_topics
            output_base = Path(self.var_export_dir.get())
            self._log_ts(f"Exportando temas: {', '.join(str(t) for t in selected_topics)}\n")

            from models.question import Question
            if not DB_PATH.exists():
                self._log_ts("Error: No se encontro la base de datos.\n")
                return

            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row

            # Build query with topic filter
            placeholders = ",".join("?" for _ in selected_topics)
            try:
                rows = conn.execute(
                    f"SELECT * FROM questions_with_options WHERE topic IN ({placeholders}) ORDER BY topic, id",
                    selected_topics,
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    f"SELECT * FROM questions WHERE topic IN ({placeholders}) ORDER BY topic, id",
                    selected_topics,
                ).fetchall()
            conn.close()

            if not rows:
                self._log_ts("No hay preguntas para los temas seleccionados.\n")
                return

            questions = []
            for row in rows:
                d = dict(row)
                q = Question(
                    id=d.get("id"), academy=d.get("academy_id", d.get("academy", 1)),
                    topic=d.get("topic", 0), question=d.get("question", ""),
                    answer1=d.get("answer1", ""), answer2=d.get("answer2", ""),
                    answer3=d.get("answer3", ""), answer4=d.get("answer4"),
                    solution=d.get("solution", 1), tip=d.get("tip"), article=d.get("article"),
                    faithfulness_score=d.get("faithfulness_score"), relevancy_score=d.get("relevancy_score"),
                    llm_model=d.get("llm_model"), difficulty=d.get("difficulty"),
                    needs_manual_review=bool(d.get("needs_manual_review", False)),
                    review_comment=d.get("review_comment"), generation_time=d.get("generation_time"),
                    review_time=d.get("review_time"),
                )
                questions.append(q)

            self._log_ts(f"{len(questions)} preguntas extraidas.\n")

            excel_dir = output_base / "excels"
            pdf_dir = output_base / "pdfs"

            if fmt in ("excel", "both"):
                self._log_ts(f"\nGenerando Excel en {excel_dir}...\n")
                from agents.agent_f_excel_generator import ExcelGeneratorAgent, ExcelFormatEnum
                meta = ExcelGeneratorAgent().generate_excels(
                    questions=questions, output_dir=excel_dir,
                    excel_format=ExcelFormatEnum(self.var_excel_fmt.get()),
                )
                for m in meta:
                    self._log_ts(f"  {m.file_name} ({m.total_questions} preg.)\n")

            if fmt in ("pdf", "both"):
                self._log_ts(f"\nGenerando PDF en {pdf_dir}...\n")
                from agents.agent_e_pdf_generator import PDFGeneratorAgent, PDFFormatEnum
                meta = PDFGeneratorAgent().generate_pdfs(
                    questions=questions, output_dir=pdf_dir,
                    pdf_format=PDFFormatEnum(self.var_pdf_fmt.get()),
                )
                for m in meta:
                    self._log_ts(f"  {m.file_name} ({m.total_questions} preg.)\n")

            self._log_ts("\nExportacion completada.\n")
        except Exception as e:
            import traceback
            self._log_ts(f"\nError: {e}\n{traceback.format_exc()}\n")
        finally:
            self._set_running_ts(False)

    # =========================================================
    # Actions: Supabase
    # =========================================================

    def _save_supabase_env(self):
        url = self.var_supa_url.get().strip()
        key = self.var_supa_key.get().strip()
        if not url or not key:
            messagebox.showwarning("Campos vacios", "Completa URL y Key.")
            return
        set_key(str(ENV_PATH), "SUPABASE_URL", url)
        set_key(str(ENV_PATH), "SUPABASE_SERVICE_ROLE_KEY", key)
        os.environ["SUPABASE_URL"] = url
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = key
        messagebox.showinfo("Guardado", "Credenciales guardadas en .env")

    def _test_supabase(self):
        self._log_clear()
        self._log("Probando conexion a Supabase...\n")
        os.environ["SUPABASE_URL"] = self.var_supa_url.get().strip()
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = self.var_supa_key.get().strip()
        result = supabase_get("topic_type", {"select": "id", "limit": "1"})
        if result is not None:
            self._log("Conexion exitosa.\n")
            messagebox.showinfo("OK", "Conexion a Supabase exitosa.")
            # Auto-load topic types on successful connection
            self._load_topic_types()
        else:
            self._log("Error conectando a Supabase. Verifica URL y Key.\n")
            messagebox.showerror("Error", "No se pudo conectar a Supabase.")

    def _load_topic_types(self):
        os.environ["SUPABASE_URL"] = self.var_supa_url.get().strip()
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = self.var_supa_key.get().strip()

        for item in self.tt_tree.get_children():
            self.tt_tree.delete(item)
        for item in self.supa_topics_tree.get_children():
            self.supa_topics_tree.delete(item)

        self._log("Cargando topic types (Study)...\n")
        data = supabase_get("topic_type", {"level": "eq.Study", "select": "id,topic_type_name,description,academy_id"})
        if data is None:
            self._log("Error cargando topic types.\n")
            return
        self._topic_types = data
        for tt in data:
            self.tt_tree.insert("", tk.END, iid=str(tt["id"]), values=(
                tt["id"], tt["topic_type_name"], tt.get("academy_id", ""), tt.get("description", "") or ""
            ))
        self._log(f"{len(data)} topic types encontrados.\n")

    def _on_tt_select(self, event):
        sel = self.tt_tree.selection()
        if not sel:
            return
        tt_id = int(sel[0])
        for item in self.supa_topics_tree.get_children():
            self.supa_topics_tree.delete(item)

        self._log(f"Cargando topics para topic_type {tt_id}...\n")
        data = supabase_get("topic", {
            "topic_type_id": f"eq.{tt_id}",
            "select": "id,topic_name,topic_short_name,order,total_questions",
            "order": "order.asc"
        })
        if data is None:
            self._log("Error cargando topics.\n")
            return
        self._supa_topics = data
        # Build mapping order -> topic for upload use
        self._supa_topic_by_order = {t["order"]: t for t in data if t.get("order")}
        for t in data:
            self.supa_topics_tree.insert("", tk.END, values=(
                t["id"], t.get("order", ""), t.get("topic_name", ""), t.get("total_questions", 0)
            ))
        self._log(f"{len(data)} topics encontrados.\n")
        # Refresh upload list with remote counts
        self._refresh_upload_topics()

    def _refresh_upload_topics(self):
        """Refresh the upload topic list showing local vs remote question counts."""
        for item in self.upload_topic_tree.get_children():
            self.upload_topic_tree.delete(item)
        stats = get_db_stats()
        remote_by_order = getattr(self, "_supa_topic_by_order", {})
        for topic_num, local_count in sorted(stats["topics"].items()):
            remote_topic = remote_by_order.get(topic_num)
            remote_count = remote_topic.get("total_questions", 0) if remote_topic else "-"
            self.upload_topic_tree.insert(
                "", tk.END, values=(f"Tema {topic_num}", local_count, remote_count),
                iid=str(topic_num),
            )

    def _on_upload_selected(self):
        """Upload selected topics' questions from local DB to Supabase."""
        if self._running:
            messagebox.showwarning("Ocupado", "Espera a que termine el proceso actual.")
            return

        # Validate topic_type is selected
        tt_sel = self.tt_tree.selection()
        if not tt_sel:
            messagebox.showwarning("Sin topic_type", "Selecciona un Topic Type en la lista de la izquierda.")
            return

        tt_id = int(tt_sel[0])
        tt_vals = self.tt_tree.item(tt_sel[0], "values")
        tt_name = tt_vals[1] if len(tt_vals) > 1 else f"id={tt_id}"
        academy_id = int(tt_vals[2]) if len(tt_vals) > 2 and tt_vals[2] else 1

        # Validate topics selected
        sel = self.upload_topic_tree.selection()
        if not sel:
            messagebox.showwarning("Sin temas", "Selecciona al menos un tema para subir.")
            return

        selected_nums = [int(s) for s in sel]

        # Check we have remote topics mapped
        remote_map = getattr(self, "_supa_topic_by_order", {})
        unmapped = [n for n in selected_nums if n not in remote_map]
        if unmapped:
            messagebox.showerror(
                "Temas sin mapping",
                f"Los siguientes temas no tienen topic remoto en Supabase:\n"
                f"{', '.join(f'Tema {n}' for n in unmapped)}\n\n"
                f"Crea los temas primero con 'Crear temas en Supabase'."
            )
            return

        summary = "\n".join(
            f"  Tema {n} -> {remote_map[n]['topic_name'][:50]}" for n in selected_nums
        )
        if not messagebox.askyesno(
            "Subir preguntas",
            f"Subir preguntas a topic_type '{tt_name}'?\n\n{summary}\n\n"
            f"Las preguntas duplicadas (mismo texto) se omitiran automaticamente."
        ):
            return

        self._log_clear()
        self._set_running(True)
        threading.Thread(
            target=self._run_upload, args=(selected_nums, remote_map, academy_id),
            daemon=True,
        ).start()

    def _run_upload(self, selected_nums: list[int], remote_map: dict, academy_id: int):
        """Upload questions from local SQLite to Supabase using bulk_upload_questions RPC."""
        try:
            if not DB_PATH.exists():
                self._log_ts("Error: Base de datos local no encontrada.\n")
                return

            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            total_uploaded = 0
            total_skipped = 0
            total_options = 0

            for tema_num in selected_nums:
                if not self._running:
                    break

                topic = remote_map[tema_num]
                topic_id = topic["id"]
                self._log_ts(f"\n{'='*50}\n  Tema {tema_num} -> {topic['topic_name'][:50]}\n{'='*50}\n")

                # Get local questions for this topic
                rows = conn.execute(
                    "SELECT * FROM questions WHERE topic = ? ORDER BY id",
                    (tema_num,),
                ).fetchall()

                if not rows:
                    self._log_ts("  Sin preguntas locales.\n")
                    continue

                # Get existing questions in Supabase for this topic to avoid duplicates
                self._log_ts("  Verificando duplicados...\n")
                existing = supabase_get("questions", {
                    "topic": f"eq.{topic_id}",
                    "select": "question",
                })
                existing_texts = set()
                if existing:
                    existing_texts = {q["question"].strip().lower() for q in existing}

                # Build payload for bulk_upload_questions RPC
                def _clean(text: str) -> str:
                    """Remove null bytes and other problematic Unicode chars for PostgreSQL."""
                    if not text:
                        return ""
                    return text.replace("\x00", "").strip()

                questions_payload = []
                skipped = 0

                for row in rows:
                    d = dict(row)
                    q_id = d.get("id")
                    q_text = _clean(d.get("question") or "")

                    # Skip duplicates
                    if q_text.lower() in existing_texts:
                        skipped += 1
                        continue

                    # Read options from question_options table
                    opt_rows = conn.execute(
                        "SELECT answer, is_correct, option_order FROM question_options "
                        "WHERE question_id = ? ORDER BY option_order",
                        (q_id,),
                    ).fetchall()

                    options = []
                    for opt in opt_rows:
                        od = dict(opt)
                        options.append({
                            "answer": _clean(od["answer"]),
                            "is_correct": bool(od["is_correct"]),
                            "option_order": od["option_order"],
                        })

                    questions_payload.append({
                        "question": q_text,
                        "topic_id": topic_id,
                        "academy_id": academy_id,
                        "tip": _clean(d.get("tip") or ""),
                        "article": _clean(d.get("article") or ""),
                        "published": True,
                        "order": 0,
                        "question_image_url": "",
                        "retro_image_url": "",
                        "classification_category_id": None,
                        "classification_topic_id": None,
                        "options": options,
                    })

                total_skipped += skipped

                if not questions_payload:
                    self._log_ts(f"  {skipped} duplicadas, 0 nuevas. Nada que subir.\n")
                    continue

                self._log_ts(f"  {skipped} duplicadas omitidas, subiendo {len(questions_payload)} nuevas...\n")

                # Call bulk_upload_questions RPC
                result = supabase_rpc("bulk_upload_questions", {
                    "p_questions": questions_payload,
                })

                if result is None:
                    self._log_ts("  Error: sin respuesta del servidor.\n")
                    continue

                if isinstance(result, dict) and result.get("error"):
                    self._log_ts(f"  Error: {result['error']}\n")
                    continue

                # Parse result
                if isinstance(result, dict):
                    inserted = result.get("questions_inserted", 0)
                    opts = result.get("options_inserted", 0)
                    errors = result.get("errors", [])
                    total_uploaded += inserted
                    total_options += opts
                    self._log_ts(f"  {inserted} preguntas subidas, {opts} opciones creadas\n")
                    if errors:
                        self._log_ts(f"  {len(errors)} errores:\n")
                        for err in errors[:5]:
                            self._log_ts(f"    [{err.get('index')}] {err.get('error')}\n")
                        if len(errors) > 5:
                            self._log_ts(f"    ... y {len(errors) - 5} mas\n")
                else:
                    self._log_ts(f"  Respuesta inesperada: {result}\n")

            conn.close()
            self._log_ts(
                f"\nCompletado: {total_uploaded} preguntas subidas, "
                f"{total_options} opciones, {total_skipped} duplicadas omitidas.\n"
            )

            # Refresh remote topics to update counts
            self.after(300, lambda: self._on_tt_select(None))
        except Exception as e:
            import traceback
            self._log_ts(f"\nError: {e}\n{traceback.format_exc()}\n")
        finally:
            self._set_running_ts(False)

    def _on_create_topics_supabase(self):
        """Create or update topics in Supabase for the selected topic_type."""
        if self._running:
            messagebox.showwarning("Ocupado", "Espera a que termine el proceso actual.")
            return

        # Get selected topic_type
        sel = self.tt_tree.selection()
        if not sel:
            messagebox.showwarning("Sin seleccion", "Primero carga los Topic Types y selecciona uno.")
            return
        tt_id = int(sel[0])
        tt_vals = self.tt_tree.item(sel[0], "values")
        tt_name = tt_vals[1] if len(tt_vals) > 1 else f"id={tt_id}"

        # Check titles file exists
        titles_path = PROJECT_ROOT / "scripts" / "topic_titles.json"
        if not titles_path.exists():
            messagebox.showwarning(
                "Sin titulos",
                "No se encontro scripts/topic_titles.json.\n"
                "Ejecuta primero 'Extraer titulos temas' en la pestana Agente Z."
            )
            return

        # Load titles
        with open(titles_path, encoding="utf-8") as f:
            titles = json.load(f)

        # Build mapping of existing topics by order
        existing_by_order = {}
        for child in self.supa_topics_tree.get_children():
            vals = self.supa_topics_tree.item(child, "values")
            if vals and vals[1]:
                try:
                    order = int(vals[1])
                    existing_by_order[order] = {
                        "id": int(vals[0]),
                        "name": vals[2] if len(vals) > 2 else "",
                    }
                except (ValueError, TypeError):
                    pass

        # Classify: new vs update (name changed)
        to_create = {}
        to_update = {}
        for k, title in titles.items():
            num = int(k)
            if num in existing_by_order:
                old_name = existing_by_order[num]["name"]
                if old_name != title:
                    to_update[num] = {"id": existing_by_order[num]["id"], "new_name": title, "old_name": old_name}
            else:
                to_create[num] = title

        if not to_create and not to_update:
            messagebox.showinfo("Todo actualizado", "Todos los temas ya existen con los nombres correctos.")
            return

        # Build summary
        lines = []
        if to_update:
            lines.append(f"ACTUALIZAR nombre ({len(to_update)}):")
            for n in sorted(to_update):
                lines.append(f"  Tema {n}: '{to_update[n]['old_name']}' -> '{to_update[n]['new_name']}'")
        if to_create:
            lines.append(f"\nCREAR nuevos ({len(to_create)}):")
            for n in sorted(to_create):
                lines.append(f"  Tema {n}: {to_create[n]}")

        if not messagebox.askyesno(
            "Crear/Actualizar temas",
            f"Topic type: '{tt_name}'\n\n" + "\n".join(lines)
        ):
            return

        def _run():
            try:
                # Update existing topics
                for num, info in sorted(to_update.items()):
                    result, err = supabase_patch(
                        "topic",
                        {"id": f"eq.{info['id']}"},
                        {"topic_name": info["new_name"]},
                    )
                    if err is None:
                        self._log_ts(f"  Actualizado Tema {num}: {info['new_name']}\n")
                    else:
                        self._log_ts(f"  ERROR actualizando Tema {num}: {err}\n")

                # Create new topics
                if to_create:
                    # Get academy_id from topic_type
                    tt_data = supabase_get("topic_type", {"id": f"eq.{tt_id}", "select": "academy_id"})
                    academy_id = tt_data[0]["academy_id"] if tt_data else None
                    if not academy_id:
                        self._log_ts("ERROR: no se pudo obtener academy_id del topic_type\n")
                        return

                    topics_data = []
                    for num in sorted(to_create):
                        topics_data.append({
                            "topic_type_id": tt_id,
                            "topic_name": to_create[num],
                            "topic_short_name": f"Tema {num}",
                            "order": num,
                            "academy_id": academy_id,
                            "is_premium": False,
                        })
                    created = supabase_post("topic", topics_data)
                    if created:
                        self._log_ts(f"  Creados {len(created)} temas nuevos\n")
                        for t in created:
                            self._log_ts(f"    id={t['id']}: Tema {t.get('order', '?')} - {t['topic_name']}\n")
                    else:
                        self._log_ts("  ERROR creando temas nuevos\n")

                self._log_ts("Sincronizacion completada.\n")
            except Exception as e:
                self._log_ts(f"Error: {e}\n")
            finally:
                self._set_running_ts(False)
                self.after(500, lambda: self._on_tt_select(None))

        self._log_clear()
        self._set_running(True)
        threading.Thread(target=_run, daemon=True).start()

    # =========================================================
    # Tab 6: Configuracion
    # =========================================================

    # All .env parameters grouped by section with descriptions
    ENV_SECTIONS = [
        ("OpenAI", [
            ("OPENAI_API_KEY", "str", "Clave API de OpenAI para todos los agentes"),
            ("OPENAI_MODEL", "str", "Modelo general de OpenAI (ej: gpt-5-mini)"),
        ]),
        ("Agente Z - Chunking", [
            ("CHUNK_SIZE", "int", "Tamano objetivo de cada chunk en caracteres"),
            ("CHUNK_OVERLAP", "int", "Solapamiento entre chunks contiguos (chars)"),
            ("MAX_MERGE_STEPS", "int", "Maximo de fusiones por chunk durante coordinacion"),
            ("MAX_MERGED_CHARS", "int", "Limite duro de caracteres tras fusiones"),
        ]),
        ("Agente C - Calidad", [
            ("QUALITY_THRESHOLD_FAITHFULNESS", "float", "Umbral de fidelidad para aprobar (0-1). Preguntas con score >= este valor se aprueban automaticamente"),
            ("QUALITY_THRESHOLD_RELEVANCY", "float", "Umbral de relevancia para aprobar (0-1). Preguntas con score >= este valor se aprueban automaticamente"),
            ("QUALITY_THRESHOLD_AUTO_FAIL_FAITHFULNESS", "float", "Umbral de rechazo automatico en fidelidad (0-1). Por debajo se rechazan sin retry"),
            ("QUALITY_THRESHOLD_AUTO_FAIL_RELEVANCY", "float", "Umbral de rechazo automatico en relevancia (0-1). Por debajo se rechazan sin retry"),
            ("AGENT_C_DIFFICULTY_MODEL", "str", "Modelo para analisis de dificultad de preguntas"),
        ]),
        ("Reintentos", [
            ("MAX_RETRIES", "int", "Numero maximo de reintentos al generar una pregunta que no pasa calidad"),
        ]),
        ("Agente B - Generacion", [
            ("BATCH_SIZE", "int", "Preguntas por chunk (legacy, usar NUM_QUESTIONS_PER_CHUNK)"),
            ("NUM_QUESTIONS_PER_CHUNK", "int", "Numero de preguntas a generar por cada chunk"),
            ("NUM_ANSWER_OPTIONS", "int", "Numero de opciones de respuesta (max: 4)"),
            ("NUM_CORRECT_OPTIONS", "int", "Opciones correctas esperadas (actualmente: 1)"),
            ("GENERATION_MODEL", "str", "Modelo de OpenAI para generar preguntas"),
            ("GENERATION_MAX_TOKENS", "int", "Maximo tokens de salida del modelo generador"),
            ("GENERATION_REASONING_EFFORT", "str", "Esfuerzo de razonamiento GPT-5: low | medium | high"),
            ("TIP_MIN_WORDS", "int", "Palabras minimas en la pista/tip de la pregunta"),
            ("TIP_MAX_WORDS", "int", "Palabras maximas en la pista/tip de la pregunta"),
        ]),
        ("Agente B - Comportamiento", [
            ("B_ENABLE_GUARDIAN", "bool", "Activar guardian (paso de evaluacion interna)"),
            ("B_ENABLE_SEQUENTIAL_EXPANSION", "bool", "Permitir expansion secuencial (chunks prev/next)"),
            ("B_ENABLE_SEMANTIC_EXPANSION", "bool", "Permitir expansion semantica (FAISS)"),
            ("B_ENABLE_MULTI_CHUNK", "bool", "Toggle global para multi-chunk"),
            ("B_MAX_EXPANSIONS", "int", "Maximo de expansiones por chunk"),
        ]),
        ("Pipeline BCDE", [
            ("BCDE_QUESTIONS_PER_CHUNK", "int", "Preguntas por chunk en el pipeline BCDE"),
            ("BCDE_CHUNK_CACHE_PATH", "str", "Ruta al cache de chunks usados (vacio = auto)"),
        ]),
        ("Deduplicacion", [
            ("SIMILARITY_THRESHOLD", "float", "Umbral de similitud para detectar duplicados (0-1)"),
        ]),
        ("Agente Z - Rewriter", [
            ("REWRITER_MODEL", "str", "Modelo de OpenAI para reescribir chunks"),
            ("REWRITER_TEMPERATURE", "float", "Temperatura del modelo (0.0 = deterministico)"),
            ("REWRITER_MAX_TOKENS", "int", "Max tokens de salida para reescritura"),
            ("REWRITER_MAX_INPUT_CHARS", "int", "Max chars de entrada antes de auto-split"),
            ("REWRITER_CONCURRENCY", "int", "Workers paralelos por documento (reducir si rate limit)"),
            ("REWRITER_TIMEOUT_SECONDS", "int", "Timeout sin progreso antes de abortar tarea (seg)"),
            ("REWRITER_DOC_TIMEOUT_SECONDS", "int", "Timeout duro por documento (0 = desactivado)"),
            ("AGENT_LLM_TIMEOUT_SECONDS", "int", "Timeout para llamadas LLM del Agente Z (seg)"),
            ("AGENT_REASONING_EFFORT", "str", "Esfuerzo razonamiento Agente Z: low | medium | high"),
            ("REWRITER_REASONING_EFFORT", "str", "Esfuerzo razonamiento reescritura: low | medium | high"),
            ("REWRITER_FORCE_REWRITE", "bool", "Forzar reescritura de todos los chunks"),
            ("REWRITER_PARTIAL_SAVE_EVERY", "int", "Guardar cache parcial cada N chunks (0 = desactivar)"),
            ("REWRITER_COORDINATION_MODE", "str", "Modo coordinacion: llm | heuristic"),
            ("REWRITER_COORDINATION_CACHE_ENABLE", "bool", "Activar cache de coordinacion"),
            ("REWRITER_COORDINATION_CACHE_EVERY", "int", "Guardar cache coordinacion cada N pasos"),
        ]),
        ("Pattern Learning", [
            ("REWRITER_PATTERN_ENABLE", "bool", "Activar aprendizaje de patrones regex"),
            ("REWRITER_PATTERN_MIN_HITS", "int", "Hits minimos antes de aplicar patron"),
            ("REWRITER_PATTERN_MIN_LEN", "int", "Longitud minima de linea para patron"),
            ("REWRITER_PATTERN_MAX_LEN", "int", "Longitud maxima de linea para patron"),
            ("REWRITER_PATTERN_BOUNDARY_LINES", "int", "Lineas frontera escaneadas para aprendizaje"),
            ("REWRITER_PATTERN_PATH", "str", "Ruta al JSON de registro de patrones (vacio = auto)"),
        ]),
        ("Supabase", [
            ("SUPABASE_URL", "str", "URL del proyecto Supabase"),
            ("SUPABASE_SERVICE_ROLE_KEY", "str", "Clave Service Role de Supabase"),
        ]),
    ]

    def _build_config_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Configuracion ")

        # Scrollable canvas
        canvas = tk.Canvas(tab, highlightthickness=0)
        vscroll = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Stretch inner frame to canvas width
            canvas.itemconfig(canvas.find_all()[0], width=canvas.winfo_width())

        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_configure)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120 or (-1 if event.num == 5 else 1)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add=True)

        self._config_vars: dict[str, tk.Variable] = {}

        for section_name, params in self.ENV_SECTIONS:
            frame = ttk.LabelFrame(inner, text=section_name)
            frame.pack(fill=tk.X, padx=10, pady=6)

            for row_idx, (key, dtype, desc) in enumerate(params):
                current = os.getenv(key, "")

                # Label (key name)
                lbl = ttk.Label(frame, text=key, font=("Menlo", 10, "bold"), width=38, anchor=tk.W)
                lbl.grid(row=row_idx * 2, column=0, sticky=tk.W, padx=8, pady=(6, 0))

                # Input widget
                if dtype == "bool":
                    var = tk.BooleanVar(value=current.lower() in ("true", "1", "yes"))
                    ttk.Checkbutton(frame, text="Activado", variable=var).grid(
                        row=row_idx * 2, column=1, sticky=tk.W, padx=8, pady=(6, 0)
                    )
                elif key in ("OPENAI_API_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
                    var = tk.StringVar(value=current)
                    ttk.Entry(frame, textvariable=var, width=50, show="*").grid(
                        row=row_idx * 2, column=1, sticky=tk.EW, padx=8, pady=(6, 0)
                    )
                else:
                    var = tk.StringVar(value=current)
                    ttk.Entry(frame, textvariable=var, width=50).grid(
                        row=row_idx * 2, column=1, sticky=tk.EW, padx=8, pady=(6, 0)
                    )

                self._config_vars[key] = var

                # Description
                ttk.Label(frame, text=desc, foreground="gray", wraplength=600, font=("Helvetica", 9)).grid(
                    row=row_idx * 2 + 1, column=0, columnspan=2, sticky=tk.W, padx=24, pady=(0, 4)
                )

            frame.columnconfigure(1, weight=1)

        # Save button at the bottom
        btn_frame = ttk.Frame(inner)
        btn_frame.pack(fill=tk.X, padx=10, pady=12)
        ttk.Button(btn_frame, text="Guardar configuracion en .env", style="Run.TButton",
                   command=self._save_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Recargar desde .env", command=self._reload_config).pack(side=tk.LEFT, padx=4)

    def _save_config(self):
        """Save all config vars to .env file."""
        changed = 0
        for key, var in self._config_vars.items():
            if isinstance(var, tk.BooleanVar):
                val = "true" if var.get() else "false"
            else:
                val = var.get()
            old = os.getenv(key, "")
            if val != old:
                set_key(str(ENV_PATH), key, val)
                os.environ[key] = val
                changed += 1
        if changed:
            self._log(f"Guardados {changed} parametros en .env\n")
            messagebox.showinfo("Guardado", f"{changed} parametros actualizados en .env")
        else:
            messagebox.showinfo("Sin cambios", "No hubo cambios que guardar.")

    def _reload_config(self):
        """Reload values from .env into the config tab."""
        load_dotenv(ENV_PATH, override=True)
        for key, var in self._config_vars.items():
            val = os.getenv(key, "")
            if isinstance(var, tk.BooleanVar):
                var.set(val.lower() in ("true", "1", "yes"))
            else:
                var.set(val)
        self._log("Configuracion recargada desde .env\n")

    # =========================================================
    # Close
    # =========================================================

    def _on_close(self):
        if self._running:
            if not messagebox.askyesno("Proceso activo", "Hay un proceso en ejecucion. Salir?"):
                return
            self._on_stop()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()