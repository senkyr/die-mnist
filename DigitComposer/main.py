import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import argparse
import os
import glob
import shutil
import sys
import zipfile
import tempfile
import atexit
import random
from pathlib import Path

# Kořen repozitáře do sys.path kvůli sdílenému balíčku die_common.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from die_common.data import allocate_counts  # noqa: E402
from die_common.paths import COMPOSED_DIR, DATA_DIR  # noqa: E402

# Dark UI palette matching DigitCollector
BG_COLOR = "#2b2b2b"
FG_COLOR = "#d8dde6"
CANVAS_BORDER = "#43464a"
BUTTON_BG = "#3b3f42"
BUTTON_FG = "#e6e9ef"
BUTTON_ACTIVE_BG = "#4a4f53"
BUTTON_ACTIVE_FG = "#ffffff"
BUTTON_BORDER = "#5a5d60"
LISTBOX_BG = "#1e1e1e"
LISTBOX_FG = "#d8dde6"
LISTBOX_SELECT_BG = "#4a9eff"
LISTBOX_SELECT_FG = "#ffffff"

MAX_SOURCES = 99
MAX_SAMPLES_PER_SOURCE = 9999
NUM_CLASSES = 10
SPLIT_NAMES = ("train", "val", "test")
# Co smí ležet ve výstupní složce, aby ji Composer směl vyčistit.
COMPOSED_ENTRIES = set(SPLIT_NAMES) | {"README.md"}


def _is_within(path, parent):
    """True, pokud `path` leží v `parent` (nebo je to tatáž složka)."""
    path = os.path.normcase(os.path.abspath(path))
    parent = os.path.normcase(os.path.abspath(parent))
    try:
        return os.path.commonpath([path, parent]) == parent
    except ValueError:  # jiné disky na Windows
        return False


def _same_dir(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def prepare_default_output_dir(composed_dir=None):
    """Založí výchozí výstup shared/data/composed a vrátí ho jako start dialogu.

    V čerstvém klonu složka neexistuje (verzuje se jen shared/data/README.md),
    dialog proto začínal v shared/data, kam Composer zapisovat nesmí, a data
    končila ve složce, kterou Learner ani Tester nečtou.
    """
    if composed_dir is None:
        composed_dir = COMPOSED_DIR
    os.makedirs(composed_dir, exist_ok=True)
    return str(composed_dir)


def lowest_free_source_id(used_ids, max_id=MAX_SOURCES):
    """Nejnižší volné ID zdroje 1..max_id – ID je předpona SS v názvu SSOOOO.bmp.

    Počítadlo, které jen rostlo, dávalo po víc než 99 přidáních (i s odebíráním)
    třímístná ID a sedmimístné názvy souborů. Odebraný zdroj ID uvolní.
    """
    used = set(used_ids)
    for source_id in range(1, max_id + 1):
        if source_id not in used:
            return source_id
    raise ValueError(f"Všech {max_id} ID zdrojů je obsazeno. Odeberte některý zdroj.")


class DataSource:
    """Represents a single data source (directory or ZIP archive)."""

    def __init__(self, source_id, source_type, display_name, data_path, temp_dir=None):
        self.id = source_id
        self.type = source_type  # 'dir' or 'zip'
        self.display_name = display_name
        self.data_path = data_path
        self.temp_dir = temp_dir
        self.stats = self._compute_stats()

    def _compute_stats(self):
        """Count BMP files in each class subdirectory."""
        stats = {}
        for class_idx in range(NUM_CLASSES):
            class_dir = os.path.join(self.data_path, str(class_idx))
            if os.path.isdir(class_dir):
                bmp_files = glob.glob(os.path.join(class_dir, "*.bmp"))
                stats[class_idx] = len(bmp_files)
            else:
                stats[class_idx] = 0
        return stats

    def get_total_samples(self):
        """Return total number of samples across all classes."""
        return sum(self.stats.values())

    def cleanup(self):
        """Remove temporary directory if this source was extracted from ZIP."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temp dir {self.temp_dir}: {e}")


class DigitComposerApp:
    def __init__(self, master):
        self.master = master
        master.title("Digit Composer")
        master.geometry("750x700")
        master.config(padx=15, bg=BG_COLOR)
        # Add top padding manually
        tk.Frame(master, height=15, bg=BG_COLOR).pack()

        self.sources = []
        # Zapsal Composer v tomto spuštění dataset do shared/data/composed? (--pipeline)
        self.composed_written = False

        # Split configuration
        self.train_ratio = tk.DoubleVar(value=0.7)
        self.val_ratio = tk.DoubleVar(value=0.2)
        self.test_ratio = tk.DoubleVar(value=0.1)
        self.seed_var = tk.IntVar(value=42)

        # Register cleanup on exit
        atexit.register(self.cleanup_all_sources)

        self._build_ui()

    def _build_ui(self):
        """Build the main UI."""
        # Title
        title_label = tk.Label(
            self.master,
            text="Digit Composer",
            font=("Arial", 24, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        title_label.pack(pady=(0, 10))

        # Instructions
        instr_label = tk.Label(
            self.master,
            text="Sloučení a rozdělení datasetů z různých instancí DigitCollector",
            font=("Arial", 12),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        instr_label.pack(pady=(0, 15))

        # Sources frame
        sources_frame = tk.Frame(self.master, bg=BG_COLOR)
        sources_frame.pack(fill=tk.X, pady=(0, 10))

        sources_label = tk.Label(
            sources_frame,
            text="Zdroje dat",
            font=("Arial", 14),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        sources_label.pack(anchor="w", pady=(0, 5))

        # Listbox with scrollbar (fixed height)
        listbox_frame = tk.Frame(sources_frame, bg=BG_COLOR)
        listbox_frame.pack(fill=tk.X)

        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.sources_listbox = tk.Listbox(
            listbox_frame,
            bg=LISTBOX_BG,
            fg=LISTBOX_FG,
            selectbackground=LISTBOX_SELECT_BG,
            selectforeground=LISTBOX_SELECT_FG,
            font=("Courier", 10),
            yscrollcommand=scrollbar.set,
            highlightthickness=1,
            highlightbackground=CANVAS_BORDER,
            bd=0,
            height=8,
        )
        self.sources_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.config(command=self.sources_listbox.yview)

        # Buttons for source management
        btn_frame = tk.Frame(self.master, bg=BG_COLOR)
        btn_frame.pack(pady=(10, 10))

        btn_width = 16  # Width for all buttons (based on longest text)

        self.add_dir_btn = self._create_button(btn_frame, "Přidat adresář", self.add_directory, width=btn_width)
        self.add_dir_btn.pack(side=tk.LEFT, padx=5)

        self.add_zip_btn = self._create_button(btn_frame, "Přidat ZIP", self.add_zip, width=btn_width)
        self.add_zip_btn.pack(side=tk.LEFT, padx=5)

        self.remove_btn = self._create_button(btn_frame, "Odebrat", self.remove_source, width=btn_width)
        self.remove_btn.pack(side=tk.LEFT, padx=5)

        self.preview_btn = self._create_button(btn_frame, "Náhled vzorků", self.show_preview, width=btn_width)
        self.preview_btn.pack(side=tk.LEFT, padx=5)

        # Split configuration
        split_frame = tk.Frame(self.master, bg=BG_COLOR)
        split_frame.pack(fill=tk.X, pady=(10, 10))

        split_label = tk.Label(
            split_frame,
            text="Rozdělení datasetu",
            font=("Arial", 14),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        split_label.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        # Train slider
        tk.Label(split_frame, text="Train:", font=("Arial", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=1, column=0, sticky="w", padx=(0, 10))
        self.train_scale = tk.Scale(
            split_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.train_ratio,
            bg=BG_COLOR,
            fg=FG_COLOR,
            highlightthickness=0,
            troughcolor=LISTBOX_BG,
            command=self._on_ratio_change,
            length=200,
        )
        self.train_scale.grid(row=1, column=1, sticky="ew", padx=(0, 10))
        self.train_count_label = tk.Label(split_frame, text="70% (0)", font=("Arial", 10), bg=BG_COLOR, fg=FG_COLOR, width=15, anchor="w")
        self.train_count_label.grid(row=1, column=2, sticky="w")

        # Val slider
        tk.Label(split_frame, text="Val:", font=("Arial", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=2, column=0, sticky="w", padx=(0, 10))
        self.val_scale = tk.Scale(
            split_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.val_ratio,
            bg=BG_COLOR,
            fg=FG_COLOR,
            highlightthickness=0,
            troughcolor=LISTBOX_BG,
            command=self._on_ratio_change,
            length=200,
        )
        self.val_scale.grid(row=2, column=1, sticky="ew", padx=(0, 10))
        self.val_count_label = tk.Label(split_frame, text="20% (0)", font=("Arial", 10), bg=BG_COLOR, fg=FG_COLOR, width=15, anchor="w")
        self.val_count_label.grid(row=2, column=2, sticky="w")

        # Test slider
        tk.Label(split_frame, text="Test:", font=("Arial", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=3, column=0, sticky="w", padx=(0, 10))
        self.test_scale = tk.Scale(
            split_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.test_ratio,
            bg=BG_COLOR,
            fg=FG_COLOR,
            highlightthickness=0,
            troughcolor=LISTBOX_BG,
            command=self._on_ratio_change,
            length=200,
        )
        self.test_scale.grid(row=3, column=1, sticky="ew", padx=(0, 10))
        self.test_count_label = tk.Label(split_frame, text="10% (0)", font=("Arial", 10), bg=BG_COLOR, fg=FG_COLOR, width=15, anchor="w")
        self.test_count_label.grid(row=3, column=2, sticky="w")

        # Validation indicator
        self.validation_label = tk.Label(
            split_frame,
            text="✓ Součet: 100%",
            font=("Arial", 10, "bold"),
            bg=BG_COLOR,
            fg="#4a9eff",
        )
        self.validation_label.grid(row=4, column=1, columnspan=2, sticky="e", pady=(5, 10))

        # Seed slider (visually separated from ratio sliders)
        tk.Label(split_frame, text="Seed:", font=("Arial", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=5, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        self.seed_scale = tk.Scale(
            split_frame,
            from_=1,
            to=100,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.seed_var,
            bg=BG_COLOR,
            fg=FG_COLOR,
            highlightthickness=0,
            troughcolor=LISTBOX_BG,
            length=200,
        )
        self.seed_scale.grid(row=5, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))
        self.seed_value_label = tk.Label(split_frame, text="(reprodukce)", font=("Arial", 10), bg=BG_COLOR, fg=FG_COLOR, width=15, anchor="w")
        self.seed_value_label.grid(row=5, column=2, sticky="w", pady=(10, 0))

        split_frame.columnconfigure(1, weight=1)

        # Visual separator before compose button (dark)
        separator = tk.Frame(self.master, height=2, bg=CANVAS_BORDER)
        separator.pack(fill=tk.X, pady=(15, 0))

        # Compose button with padding = button height above and below
        self.compose_btn = self._create_button(self.master, "Komponovat a uložit", self.compose_dataset, width=25, height=2)
        self.compose_btn.pack(pady=(30, 30))

    def _create_button(self, parent, text, command, width=15, height=None):
        """Create a styled button."""
        btn_config = {
            "text": text,
            "command": command,
            "width": width,
            "bg": BUTTON_BG,
            "fg": BUTTON_FG,
            "activebackground": BUTTON_ACTIVE_BG,
            "activeforeground": BUTTON_ACTIVE_FG,
            "relief": "flat",
            "bd": 0,
            "highlightthickness": 1,
            "highlightbackground": BUTTON_BORDER,
            "highlightcolor": BUTTON_BORDER,
            "font": ("Arial", 11),
        }
        if height is not None:
            btn_config["height"] = height

        btn = tk.Button(parent, **btn_config)
        return btn

    def add_directory(self):
        """Add a directory as a data source."""
        if len(self.sources) >= MAX_SOURCES:
            messagebox.showerror("Chyba", f"Maximální počet zdrojů je {MAX_SOURCES}.")
            return

        # Smart initial directory: try shared/data, then parent
        program_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(program_dir)

        initial_dir = str(DATA_DIR) if DATA_DIR.is_dir() else parent_dir

        dir_path = filedialog.askdirectory(
            title="Vyberte adresář s daty",
            initialdir=initial_dir
        )
        if not dir_path:
            return

        data_paths = self._confirm_datasets(dir_path)
        if data_paths is None:
            return
        if not data_paths:
            messagebox.showerror(
                "Neplatná struktura",
                "Adresář neobsahuje očekávanou strukturu dat (0/, 1/, ... 9/, digits/0/, ... "
                "nebo train/val/test s podsložkami 0-9)."
            )
            return
        if len(self.sources) + len(data_paths) > MAX_SOURCES:
            messagebox.showerror("Chyba", f"Maximální počet zdrojů je {MAX_SOURCES}.")
            return

        for data_path in data_paths:
            source = DataSource(
                source_id=self._next_source_id(),
                source_type='dir',
                display_name=os.path.basename(dir_path),
                data_path=data_path,
            )
            self._add_source(source)

    def add_zip(self):
        """Add a ZIP archive as a data source."""
        if len(self.sources) >= MAX_SOURCES:
            messagebox.showerror("Chyba", f"Maximální počet zdrojů je {MAX_SOURCES}.")
            return

        # Smart initial directory: parent directory to see all modules
        program_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(program_dir)

        zip_path = filedialog.askopenfilename(
            title="Vyberte ZIP archiv",
            initialdir=parent_dir,
            filetypes=[("ZIP archiv", "*.zip"), ("Všechny soubory", "*.*")]
        )
        if not zip_path:
            return

        # Extract to temporary directory
        temp_dir = tempfile.mkdtemp(prefix="digitcomposer_")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            messagebox.showerror("Chyba", f"Nelze rozbalit ZIP archiv: {e}")
            return

        data_paths = self._confirm_datasets(temp_dir)
        if data_paths is None:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return
        if not data_paths or len(self.sources) + len(data_paths) > MAX_SOURCES:
            shutil.rmtree(temp_dir, ignore_errors=True)
            messagebox.showerror(
                "Neplatná struktura" if not data_paths else "Chyba",
                "ZIP archiv neobsahuje očekávanou strukturu dat." if not data_paths
                else f"Maximální počet zdrojů je {MAX_SOURCES}."
            )
            return

        for data_path in data_paths:
            source = DataSource(
                source_id=self._next_source_id(),
                source_type='zip',
                display_name=os.path.basename(zip_path),
                data_path=data_path,
                temp_dir=temp_dir,
            )
            self._add_source(source)

    def _dataset_roots(self, root_path):
        """Všechny složky s podsložkami 0-9 do hloubky 3; do nalezené se dál nesestupuje.

        Podporuje 0/…9/ přímo v kořeni, digits/0/…, jmeno/digits/0/… i splity
        train/val/test. Dřív se vracela jen první nalezená složka a další
        datasety ve vybrané složce (např. synthetic/ vedle collected/ při výběru
        shared/data, nebo data dalších žáků v jednom ZIPu) se tiše vynechaly.
        """
        found = []
        for root, dirs, files in os.walk(root_path):
            dirs.sort()
            depth = root[len(root_path):].count(os.sep)
            if self._has_class_dirs(root):
                found.append(root)
                dirs[:] = []
            elif depth >= 3:
                dirs[:] = []
        return found

    def _dataset_groups(self, root_path):
        """{dataset: [složky zdrojů]}; splity train/val/test patří k jednomu datasetu.

        Dataset rozdělený na train/val/test (DigitDreamer --split, starší
        composed) dává tři zdroje v pořadí train, val, test. Výstup Composeru
        (shared/data/composed) vnořený ve vybrané složce se přeskočí – jsou to
        odvozená data a zdroj ve výstupní složce by komponování stejně odmítlo.
        """
        groups = {}
        for path in self._dataset_roots(root_path):
            name = os.path.basename(path)
            key = os.path.dirname(path) if name in SPLIT_NAMES else path
            groups.setdefault(key, []).append(path)
        for key in list(groups):
            if _same_dir(key, COMPOSED_DIR) and not _same_dir(key, root_path):
                del groups[key]
        for key, paths in groups.items():
            order = {os.path.join(key, n): i for i, n in enumerate(SPLIT_NAMES)}
            paths.sort(key=lambda p: order.get(p, -1))
        return groups

    def _find_data_roots(self, root_path):
        """Složky zdrojů ze všech datasetů ve vybrané cestě (viz _dataset_groups)."""
        return [p for paths in self._dataset_groups(root_path).values() for p in paths]

    def _confirm_datasets(self, root_path):
        """Složky zdrojů k přidání; u více datasetů se zeptá. None = uživatel odmítl, [] = nic nenalezeno."""
        groups = self._dataset_groups(root_path)
        if len(groups) > 1:
            names = "\n".join(f"- {os.path.relpath(k, root_path)}" for k in groups)
            if not messagebox.askyesno(
                "Více datasetů",
                f"Vybraná složka obsahuje {len(groups)} datasety:\n{names}\n\n"
                "Přidat je všechny jako samostatné zdroje?",
            ):
                return None
        return [p for paths in groups.values() for p in paths]

    def _has_class_dirs(self, path):
        """Check if path contains subdirectories named 0-9."""
        if not os.path.isdir(path):
            return False

        found_classes = 0
        for class_idx in range(NUM_CLASSES):
            class_dir = os.path.join(path, str(class_idx))
            if os.path.isdir(class_dir):
                found_classes += 1

        # Require at least 5 class directories to be considered valid
        return found_classes >= 5

    def _next_source_id(self):
        return lowest_free_source_id(s.id for s in self.sources)

    def _add_source(self, source):
        """Add a source to the list and update UI."""
        self.sources.append(source)

        # Update listbox with inline statistics
        stats_str = " | ".join(f"{source.stats[i]:02d}" for i in range(NUM_CLASSES))
        display_text = f"[{source.id:02d}] {source.data_path}  {source.get_total_samples():4} vzorků --> | {stats_str} |"
        self.sources_listbox.insert(tk.END, display_text)

        # Update split counts
        self._update_split_counts()

    def remove_source(self):
        """Remove selected source from the list."""
        selection = self.sources_listbox.curselection()
        if not selection:
            messagebox.showwarning("Upozornění", "Vyberte zdroj k odebrání.")
            return

        idx = selection[0]
        source = self.sources[idx]

        # Remove from list; temp dir smažu, až ho nepoužívá žádný jiný zdroj ze stejného ZIPu
        del self.sources[idx]
        if not any(s.temp_dir and s.temp_dir == source.temp_dir for s in self.sources):
            source.cleanup()
        self.sources_listbox.delete(idx)

        # Update split counts
        self._update_split_counts()

    def _on_ratio_change(self, *args):
        """Called when any ratio slider changes."""
        self._update_split_counts()
        self._validate_ratios()

    def _validate_ratios(self):
        """Validate that ratios sum to approximately 1.0."""
        total = self.train_ratio.get() + self.val_ratio.get() + self.test_ratio.get()
        is_valid = abs(total - 1.0) < 0.01

        if is_valid:
            self.validation_label.config(
                text=f"✓ Součet: {total*100:.0f}%",
                fg="#4a9eff",
            )
        else:
            self.validation_label.config(
                text=f"⚠ Součet: {total*100:.0f}% (musí být 100%)",
                fg="#ff6b6b",
            )

    def _ratios(self):
        return (self.train_ratio.get(), self.val_ratio.get(), self.test_ratio.get())

    def _update_split_counts(self):
        """Update the displayed sample counts for each split."""
        # Stejný výpočet jako při komponování: po třídách, největší zbytek.
        train_count = val_count = test_count = 0
        ratios = self._ratios()
        if sum(ratios) > 0:
            for class_idx in range(NUM_CLASSES):
                n = sum(source.stats[class_idx] for source in self.sources)
                tr, va, te = allocate_counts(n, ratios)
                train_count += tr
                val_count += va
                test_count += te

        self.train_count_label.config(text=f"{self.train_ratio.get()*100:.0f}% ({train_count} vzorků)")
        self.val_count_label.config(text=f"{self.val_ratio.get()*100:.0f}% ({val_count} vzorků)")
        self.test_count_label.config(text=f"{self.test_ratio.get()*100:.0f}% ({test_count} vzorků)")

    def show_preview(self):
        """Open a window showing random samples from each source."""
        if not self.sources:
            messagebox.showwarning("Upozornění", "Nejprve přidejte nějaké zdroje.")
            return

        preview_window = tk.Toplevel(self.master)
        preview_window.title("Náhled vzorků")
        preview_window.config(bg=BG_COLOR, padx=10, pady=10)

        # Create grid: rows = sources, columns = classes
        SAMPLE_SIZE = 64

        # Header row with class labels
        tk.Label(preview_window, text="", bg=BG_COLOR).grid(row=0, column=0)
        for class_idx in range(NUM_CLASSES):
            lbl = tk.Label(
                preview_window,
                text=str(class_idx),
                font=("Arial", 12, "bold"),
                bg=BG_COLOR,
                fg=FG_COLOR,
            )
            lbl.grid(row=0, column=class_idx + 1, padx=2, pady=2)

        # Rows for each source
        for row_idx, source in enumerate(self.sources, start=1):
            # Source label
            src_lbl = tk.Label(
                preview_window,
                text=f"[{source.id:02d}]",
                font=("Courier", 10),
                bg=BG_COLOR,
                fg=FG_COLOR,
            )
            src_lbl.grid(row=row_idx, column=0, padx=5, pady=2)

            # Samples for each class
            for class_idx in range(NUM_CLASSES):
                class_dir = os.path.join(source.data_path, str(class_idx))
                bmp_files = glob.glob(os.path.join(class_dir, "*.bmp"))

                if bmp_files:
                    # Pick a random sample
                    sample_path = random.choice(bmp_files)

                    try:
                        img = Image.open(sample_path)
                        img = img.resize((SAMPLE_SIZE, SAMPLE_SIZE), Image.NEAREST)
                        photo = ImageTk.PhotoImage(img)

                        canvas = tk.Canvas(
                            preview_window,
                            width=SAMPLE_SIZE,
                            height=SAMPLE_SIZE,
                            bg="black",
                            highlightthickness=1,
                            highlightbackground=CANVAS_BORDER,
                        )
                        canvas.create_image(0, 0, image=photo, anchor="nw")
                        canvas.image = photo  # Keep reference
                        canvas.grid(row=row_idx, column=class_idx + 1, padx=2, pady=2)
                    except Exception as e:
                        # If can't load, show empty canvas
                        canvas = tk.Canvas(
                            preview_window,
                            width=SAMPLE_SIZE,
                            height=SAMPLE_SIZE,
                            bg="black",
                            highlightthickness=1,
                            highlightbackground=CANVAS_BORDER,
                        )
                        canvas.grid(row=row_idx, column=class_idx + 1, padx=2, pady=2)
                else:
                    # No samples in this class
                    canvas = tk.Canvas(
                        preview_window,
                        width=SAMPLE_SIZE,
                        height=SAMPLE_SIZE,
                        bg="#1e1e1e",
                        highlightthickness=1,
                        highlightbackground=CANVAS_BORDER,
                    )
                    canvas.grid(row=row_idx, column=class_idx + 1, padx=2, pady=2)

    def compose_dataset(self):
        """Compose the final dataset from all sources with train/val/test split."""
        if not self.sources:
            messagebox.showwarning("Upozornění", "Nejprve přidejte nějaké zdroje.")
            return

        # Validate ratios
        total_ratio = self.train_ratio.get() + self.val_ratio.get() + self.test_ratio.get()
        if abs(total_ratio - 1.0) >= 0.01:
            messagebox.showerror("Chyba", f"Součet poměrů musí být 100% (aktuálně {total_ratio*100:.0f}%).")
            return

        # Výstup: dialog začíná v shared/data/composed, kterou čte Learner i Tester – stačí potvrdit.
        try:
            initial_dir = prepare_default_output_dir()
        except OSError as e:
            messagebox.showerror("Chyba", f"Nelze založit výchozí výstupní složku {COMPOSED_DIR}:\n{e}")
            return

        output_dir = filedialog.askdirectory(
            title="Výstupní adresář (potvrďte předvybranou složku composed)",
            initialdir=initial_dir,
        )
        if not output_dir:
            # User cancelled
            return

        # Check if user selected EXACTLY the program directory (subdirectories are OK)
        program_dir = os.path.abspath(os.path.dirname(__file__))
        output_abs = os.path.abspath(output_dir)

        if os.path.normcase(output_abs) == os.path.normcase(program_dir):
            messagebox.showerror(
                "Neplatný adresář",
                "Nelze uložit dataset přímo do složky programu.\n\n"
                f"Složka programu: {program_dir}\n\n"
                "Vyberte prosím jinou složku nebo vytvořte podsložku."
            )
            return

        # Výstup nesmí obsahovat žádný ze zdrojů – smazáním by se zdroj zničil dřív, než se přečte
        for source in self.sources:
            if _is_within(source.data_path, output_abs):
                messagebox.showerror(
                    "Neplatný adresář",
                    f"Výstupní složka obsahuje zdroj [{source.id:02d}]:\n{source.data_path}\n\n"
                    "Vyberte jinou výstupní složku."
                )
                return

        # Check if directory exists and is not empty
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            existing = set(os.listdir(output_dir))
            foreign = sorted(existing - COMPOSED_ENTRIES)
            if foreign:
                # Dialog začíná v shared/data – jedno kliknutí vedle by jinak smazalo
                # collected/ i synthetic/. Mažu jen složku, která vypadá jako composed.
                messagebox.showerror(
                    "Složka není prázdná",
                    f"Adresář '{output_dir}' obsahuje i jiná data než train/val/test:\n"
                    f"{', '.join(foreign[:5])}{' …' if len(foreign) > 5 else ''}\n\n"
                    "Z bezpečnostních důvodů ho nepřepíšu. Vyberte prázdnou složku "
                    "nebo dřívější výstup Composeru (např. shared/data/composed)."
                )
                return
            if not messagebox.askyesno(
                "Složka není prázdná",
                f"Adresář '{output_dir}' už obsahuje dataset (train/val/test).\n\n"
                f"Chcete ho odstranit a vytvořit nový?"
            ):
                return  # User cancelled
            for split in SPLIT_NAMES:
                shutil.rmtree(os.path.join(output_dir, split), ignore_errors=True)

        # Create output structure: train/val/test with 0-9 subdirectories
        for split in SPLIT_NAMES:
            for class_idx in range(NUM_CLASSES):
                os.makedirs(os.path.join(output_dir, split, str(class_idx)), exist_ok=True)

        # Progress window
        progress_window = tk.Toplevel(self.master)
        progress_window.title("Komponování datasetu")
        progress_window.config(bg=BG_COLOR, padx=20, pady=20)
        progress_window.transient(self.master)
        progress_window.grab_set()

        progress_label = tk.Label(
            progress_window,
            text="Zpracování tříd...",
            font=("Arial", 12),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        progress_label.pack(pady=(0, 10))

        progress_bar = ttk.Progressbar(
            progress_window,
            length=400,
            mode='determinate',
        )
        progress_bar.pack(pady=(0, 10))

        progress_detail = tk.Label(
            progress_window,
            text="Třída 1 / 10",
            font=("Arial", 10),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        progress_detail.pack()

        # Get seed
        seed = self.seed_var.get()
        random.seed(seed)

        # Statistics
        total_copied = {'train': 0, 'val': 0, 'test': 0}
        errors = []

        # Process each class separately (stratified split)
        for class_idx in range(NUM_CLASSES):
            progress_detail.config(text=f"Třída {class_idx + 1} / {NUM_CLASSES}")
            progress_bar['value'] = int((class_idx / NUM_CLASSES) * 100)
            progress_window.update()

            # Collect all files for this class from all sources
            all_files = []
            for source in self.sources:
                source_prefix = source.id
                class_dir = os.path.join(source.data_path, str(class_idx))
                bmp_files = sorted(glob.glob(os.path.join(class_dir, "*.bmp")))

                for bmp_file in bmp_files:
                    # Extract original number from filename
                    basename = os.path.basename(bmp_file)
                    try:
                        original_num = int(os.path.splitext(basename)[0])
                    except ValueError:
                        errors.append(f"Nelze zpracovat soubor {bmp_file} (neplatný název)")
                        continue

                    if original_num > MAX_SAMPLES_PER_SOURCE:
                        errors.append(f"Soubor {bmp_file} má číslo > {MAX_SAMPLES_PER_SOURCE}")
                        continue

                    # Generate new filename: SSOOOO.bmp
                    new_filename = f"{source_prefix:02d}{original_num:04d}.bmp"
                    all_files.append((bmp_file, new_filename))

            # Shuffle files for this class
            random.shuffle(all_files)

            # Split according to ratios (největší zbytek, ne zaokrouhlení dolů se zbytkem do testu)
            n_train, n_val, _ = allocate_counts(len(all_files), self._ratios())
            train_end = n_train
            val_end = n_train + n_val

            train_files = all_files[:train_end]
            val_files = all_files[train_end:val_end]
            test_files = all_files[val_end:]

            # Copy files to respective splits
            for split_name, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
                for src_path, new_filename in file_list:
                    dest_path = os.path.join(output_dir, split_name, str(class_idx), new_filename)
                    try:
                        shutil.copy2(src_path, dest_path)
                        total_copied[split_name] += 1
                    except Exception as e:
                        errors.append(f"Chyba při kopírování {src_path}: {e}")

        # Final progress
        progress_bar['value'] = 100
        progress_detail.config(text="Hotovo")
        progress_window.update()
        progress_window.after(500, progress_window.destroy)

        # Show results
        result_msg = (
            f"Dataset zkompilován do '{output_dir}':\n\n"
            f"Train: {total_copied['train']} vzorků\n"
            f"Val:   {total_copied['val']} vzorků\n"
            f"Test:  {total_copied['test']} vzorků\n"
            f"Celkem: {sum(total_copied.values())} vzorků\n"
            f"Seed: {seed}"
        )

        if errors:
            result_msg += f"\n\nPočet chyb: {len(errors)}"
            if len(errors) <= 5:
                result_msg += "\n" + "\n".join(errors)
            else:
                result_msg += f"\n(zobrazeno prvních 5)\n" + "\n".join(errors[:5])

        if _same_dir(output_dir, COMPOSED_DIR):
            if sum(total_copied.values()) > 0:
                self.composed_written = True
        else:
            result_msg += (
                f"\n\nPOZOR: DigitLearner a DigitTester čtou ve výchozím stavu\n{COMPOSED_DIR}.\n"
                "Pro tuto složku je spusťte s --data_dir."
            )

        messagebox.showinfo("Hotovo", result_msg)

    def cleanup_all_sources(self):
        """Cleanup all temporary directories."""
        for source in self.sources:
            source.cleanup()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DigitComposer – sloučení a rozdělení datasetů.")
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Režim start_*.bat: skončí s kódem 1, pokud do shared/data/composed nic nezapsal, "
             "aby workflow nepokračovalo tréninkem bez dat.",
    )
    return parser.parse_args(argv)


def pipeline_exit_code(pipeline, composed_written):
    return 1 if pipeline and not composed_written else 0


def main(argv=None):
    args = parse_args(argv)
    root = tk.Tk()
    app = DigitComposerApp(root)
    root.mainloop()
    code = pipeline_exit_code(args.pipeline, app.composed_written)
    if code:
        print(f"Do {COMPOSED_DIR} se v tomto spuštění nic nezapsalo, workflow nepokračuje.")
        print("Spusťte ho znovu a v Composeru potvrďte předvybranou složku composed.")
    return code


if __name__ == "__main__":
    sys.exit(main())
