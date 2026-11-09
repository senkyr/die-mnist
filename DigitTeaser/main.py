"""
DigitTeaser - Real-time klasifikátor ručně kreslených číslic.

Načte natrénovaný model z DigitLearner a umožňuje interaktivně "poškádlit"
model kreslením číslic v canvasu s okamžitým zobrazením pravděpodobností klasifikace.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import numpy as np
import torch
import json
import sys
from pathlib import Path

# Kořen repozitáře do sys.path kvůli sdílenému balíčku die_common.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from die_common.model import load_model  # noqa: E402
from die_common.paths import MODELS_DIR, ROOT, run_sample_count  # noqa: E402

# Konstanty z DigitCollector – musí sedět s tím, jak se kreslila trénovací data
# (tloušťka tahu 15 px na plátně 256 px, zmenšení LANCZOS na 32x32), jinak
# model vidí jinak tlusté tahy, než na jakých se učil.
CANVAS_SIZE = 256
IMAGE_SIZE = 32
BRUSH_WIDTH = 15

# Dark UI palette matching other tools
BG_COLOR = "#2b2b2b"
FG_COLOR = "#d8dde6"
CANVAS_BORDER = "#43464a"
BUTTON_BG = "#3b3f42"
BUTTON_FG = "#e6e9ef"
BUTTON_ACTIVE_BG = "#4a4f53"
BUTTON_ACTIVE_FG = "#ffffff"
BUTTON_BORDER = "#5a5d60"
BAR_HIGHLIGHT = "#4a9eff"
BAR_NORMAL = "#3b3f42"

class DigitTeaserApp:
    def __init__(self, master):
        self.master = master
        master.title("Digit Teaser")
        master.geometry("+50+50")
        master.config(padx=15, pady=15, bg=BG_COLOR)

        # Model state
        self.model = None
        self.device = torch.device("cpu")
        self.model_name = None

        # Canvas state
        self.last_x, self.last_y = None, None
        self.inference_job = None
        self.dirty = True  # Přepočítat predikci jen po změně kresby nebo modelu
        self.last_probs = np.ones(10) / 10.0  # Uniform distribution initially

        # Setup GUI
        self._setup_main_content()

        # Create PIL image for drawing
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        # Bind canvas events
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self._reset_last_coords)

        # Start inference loop
        self._schedule_inference()

    def _setup_main_content(self):
        """Vytvoří hlavní oblast s canvasem a probability panelem."""
        main_frame = tk.Frame(self.master, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Left section: Button + Canvas + Clear button
        canvas_frame = tk.Frame(main_frame, bg=BG_COLOR)
        canvas_frame.pack(side=tk.LEFT, padx=(0, 15))

        # Load model button
        button_frame = tk.Frame(canvas_frame, bg=BG_COLOR)
        button_frame.pack(pady=(0, 10))

        self.load_btn = self._create_button(
            button_frame, "Načíst model...", self._load_model, width=15
        )
        self.load_btn.pack()

        # Canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            bd=0,
            highlightthickness=2,
            highlightbackground=CANVAS_BORDER,
            relief="flat",
        )
        self.canvas.pack()

        # Clear button
        self.clear_btn = self._create_button(
            canvas_frame, "Smazat", self._clear_canvas, width=10
        )
        self.clear_btn.pack(pady=(10, 0))

        # Right section: Model label + Probability panel
        prob_frame = tk.Frame(main_frame, bg=BG_COLOR)
        prob_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Model label
        label_frame = tk.Frame(prob_frame, bg=BG_COLOR)
        label_frame.pack(pady=(0, 10))

        self.model_prefix_label = tk.Label(
            label_frame,
            text="Model ",
            font=("Arial", 12),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        self.model_prefix_label.pack(side=tk.LEFT)

        self.model_name_label = tk.Label(
            label_frame,
            text="žádný",
            font=("Arial", 12, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        self.model_name_label.pack(side=tk.LEFT)

        # Probability bars canvas
        self.prob_canvas = tk.Canvas(
            prob_frame,
            width=300,
            height=300,
            bg=BG_COLOR,
            bd=0,
            highlightthickness=0,
        )
        self.prob_canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize probability bars
        self._init_probability_bars()
        self._update_probability_bars(self.last_probs)

    def _init_probability_bars(self):
        """Inicializuje vizuální elementy pro probability bars."""
        self.bar_height = 25
        self.bar_spacing = 5
        self.bar_max_width = 200

    def _update_probability_bars(self, probs):
        """Aktualizuje zobrazení probability bars."""
        self.prob_canvas.delete("all")

        # Check if probabilities are uniform (empty canvas)
        is_uniform = np.std(probs) < 0.01

        if is_uniform:
            max_idx = None
        else:
            max_idx = np.argmax(probs)

        y_offset = 0

        for i in range(10):
            prob = probs[i]

            # Determine bar color
            bar_color = BAR_HIGHLIGHT if (max_idx is not None and i == max_idx) else BAR_NORMAL

            # Draw label
            self.prob_canvas.create_text(
                5, y_offset + self.bar_height // 2,
                text=f"{i}:",
                anchor="w",
                fill=FG_COLOR,
                font=("Courier", 11)
            )

            # Draw bar
            bar_width = int(prob * self.bar_max_width)
            self.prob_canvas.create_rectangle(
                30, y_offset + 5,
                30 + bar_width, y_offset + self.bar_height - 5,
                fill=bar_color,
                outline=""
            )

            # Draw percentage
            self.prob_canvas.create_text(
                30 + self.bar_max_width + 10, y_offset + self.bar_height // 2,
                text=f"{prob * 100:.1f}%",
                anchor="w",
                fill=FG_COLOR,
                font=("Courier", 10)
            )

            y_offset += self.bar_height + self.bar_spacing

    def _create_button(self, parent, text, command, width=15):
        """Vytvoří stylizované tlačítko."""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_ACTIVE_FG,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BUTTON_BORDER,
            highlightcolor=BUTTON_BORDER,
            font=("Arial", 11),
        )
        return btn

    def _load_model(self):
        """Načte model z vybrané složky."""
        # Smart initial directory: central models directory
        initial_dir = MODELS_DIR if MODELS_DIR.exists() else ROOT

        folder = filedialog.askdirectory(
            title="Vyberte složku s modelem (obsahující config.json a digit_cnn.pt)",
            initialdir=str(initial_dir)
        )

        if not folder:
            return

        folder_path = Path(folder)
        config_path = folder_path / "config.json"
        model_path = folder_path / "digit_cnn.pt"

        # Check files exist
        if not config_path.exists():
            messagebox.showerror(
                "Chyba",
                f"Složka neobsahuje config.json:\n{folder}"
            )
            return

        if not model_path.exists():
            messagebox.showerror(
                "Chyba",
                f"Složka neobsahuje digit_cnn.pt:\n{folder}"
            )
            return

        try:
            # Load config to get dropout
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            dropout = config.get('dropout', 0.2)
            num_classes = config.get('num_classes', 10)
            if num_classes != 10:
                raise ValueError(f"Model má {num_classes} tříd, DigitTeaser umí jen číslice 0-9.")

            sample_count = run_sample_count(folder_path)

            # Create model and load weights (eval mode)
            self.model = load_model(model_path, num_classes=10, dropout=dropout, device=self.device)
            self.dirty = True

            # Update UI with model name and optional sample count
            self.model_name = folder_path.name
            if sample_count is not None:
                self.model_name_label.config(text=f"{self.model_name} ({sample_count} vzorků)")
            else:
                self.model_name_label.config(text=f"{self.model_name}")

            print(f"Model successfully loaded from {folder}")
            print(f"  Dropout: {dropout}")
            if sample_count:
                print(f"  Samples: {sample_count}")

        except Exception as e:
            messagebox.showerror(
                "Chyba při načítání modelu",
                f"Nepodařilo se načíst model:\n{str(e)}"
            )
            self.model = None

    def _clear_canvas(self):
        """Vyčistí canvas a resetuje predikce."""
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        # Reset to uniform probabilities
        self.last_probs = np.ones(10) / 10.0
        self._update_probability_bars(self.last_probs)
        self.dirty = True

    def _start_draw(self, event):
        """Zahájí kreslení."""
        self.last_x, self.last_y = event.x, event.y

    def _draw_on_canvas(self, event):
        """Kreslí na canvas při pohybu myši."""
        if self.last_x is not None and self.last_y is not None:
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill="white", width=BRUSH_WIDTH, capstyle=tk.ROUND, smooth=True
            )

            # Draw on PIL image (stejně jako DigitCollector)
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255, width=BRUSH_WIDTH, joint="curve"
            )
            self.dirty = True

        self.last_x, self.last_y = event.x, event.y

    def _reset_last_coords(self, event):
        """Resetuje poslední souřadnice při uvolnění tlačítka."""
        self.last_x, self.last_y = None, None

    def _is_canvas_empty(self):
        """Kontroluje, zda je canvas prázdný."""
        arr = np.array(self.image)
        return np.sum(arr) == 0

    def _preprocess_canvas(self):
        """Předzpracuje canvas pro inference."""
        # Resize to 32x32
        resized = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        # Convert to numpy array and normalize
        arr = np.asarray(resized, dtype=np.float32) / 255.0

        # Convert to tensor [1, 1, 32, 32]
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

        return tensor

    def _inference_loop(self):
        """Hlavní inference smyčka volaná každých 50ms (počítá jen po změně)."""
        if not self.dirty:
            self._schedule_inference()
            return
        self.dirty = False
        if self.model is None:
            # No model loaded, show uniform probabilities
            self.last_probs = np.ones(10) / 10.0
            self._update_probability_bars(self.last_probs)
        elif self._is_canvas_empty():
            # Canvas is empty, show uniform probabilities
            self.last_probs = np.ones(10) / 10.0
            self._update_probability_bars(self.last_probs)
        else:
            # Run inference
            try:
                with torch.no_grad():
                    tensor = self._preprocess_canvas()
                    output = self.model(tensor)
                    probs = torch.softmax(output, dim=1).squeeze().numpy()

                    self.last_probs = probs
                    self._update_probability_bars(probs)
            except Exception as e:
                print(f"Inference error: {e}")

        # Schedule next iteration
        self._schedule_inference()

    def _schedule_inference(self):
        """Naplánuje další iteraci inference."""
        if self.inference_job is not None:
            self.master.after_cancel(self.inference_job)
        self.inference_job = self.master.after(50, self._inference_loop)


def main():
    root = tk.Tk()
    app = DigitTeaserApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
