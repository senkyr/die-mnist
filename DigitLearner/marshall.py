import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

# Kořen repozitáře do sys.path kvůli sdílenému balíčku die_common.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from die_common.config import HyperParams  # noqa: E402
from die_common.paths import LEARNER_CONFIG  # noqa: E402

CONFIG_PATH = LEARNER_CONFIG

# Výchozí hodnoty drží jediné místo – HyperParams, které čte i train.py.
DEFAULTS = HyperParams().as_dict()

PARAMS = [
    {"key": "epochs", "label": "Epochs", "type": "int", "min": 1, "max": 200, "step": 1},
    {"key": "batch_size", "label": "Batch size", "type": "int", "min": 8, "max": 256, "step": 8},
    {"key": "learning_rate", "label": "Learning rate", "type": "float", "min": 0.0001, "max": 0.01, "step": 0.0001},
    {"key": "val_ratio", "label": "Validation ratio", "type": "float", "min": 0.0, "max": 0.5, "step": 0.05},
    {"key": "test_ratio", "label": "Test ratio", "type": "float", "min": 0.0, "max": 0.5, "step": 0.05},
    {"key": "step_size", "label": "LR step size", "type": "int", "min": 1, "max": 50, "step": 1},
    {"key": "lr_gamma", "label": "LR gamma", "type": "float", "min": 0.1, "max": 1.0, "step": 0.05},
    {"key": "dropout", "label": "Dropout", "type": "float", "min": 0.0, "max": 0.8, "step": 0.05},
    {"key": "seed", "label": "Seed", "type": "int", "min": 1, "max": 100, "step": 1},
]

# DigitCollector palette
BG_COLOR = "#2b2b2b"
FG_COLOR = "#d8dde6"
CANVAS_BORDER = "#43464a"
BUTTON_BG = "#3b3f42"
BUTTON_FG = "#e6e9ef"
BUTTON_ACTIVE_BG = "#4a4f53"
BUTTON_ACTIVE_FG = "#ffffff"
BUTTON_BORDER = "#5a5d60"
SCALE_TROUGH = "#1a1a1a"


def validate_values(data: dict) -> None:
    """Stejná validace, jakou config.json projde v train.py; ValueError s popisem chyby.

    Posuvníky samy nehlídají vztahy mezi hodnotami (např. val 0,5 + test 0,5
    nenechá nic na trénink), takže bez kontroly by editor uložil konfiguraci,
    na které trénink spadne.
    """
    hp = HyperParams()
    hp.update_from_dict(data)
    hp.validate()


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                merged = DEFAULTS.copy()
                merged.update({k: data.get(k, DEFAULTS[k]) for k in DEFAULTS})
                return merged
    return DEFAULTS.copy()


class HyperparamEditor(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DigitLearner Hyperparameters")
        self.configure(bg=BG_COLOR, padx=28, pady=18)
        self.resizable(False, False)
        self.grid_columnconfigure(1, weight=1)

        self.config_data = load_config()
        self.vars: dict[str, tk.Variable] = {}
        self.value_labels: dict[str, tk.Label] = {}

        self._build_ui()
        self.bind("<Return>", lambda _: self._save_and_close())
        self.bind("<space>", lambda _: self.reset_defaults())
        self.bind("<Escape>", lambda _: self.close())

    def _build_ui(self) -> None:
        header = tk.Label(
            self,
            text=f"Config file {CONFIG_PATH}",
            font=("Segoe UI", 13, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        header.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))

        for row, meta in enumerate(PARAMS, start=1):
            key = meta["key"]
            is_int = meta["type"] == "int"

            label = tk.Label(
                self,
                text=meta["label"],
                bg=BG_COLOR,
                fg=FG_COLOR,
                anchor="w",
                font=("Segoe UI", 12),
            )
            label.grid(row=row, column=0, sticky="w", padx=(0, 16), pady=5)

            if is_int:
                var: tk.Variable = tk.IntVar(value=int(round(self.config_data[key])))
            else:
                var = tk.DoubleVar(value=float(self.config_data[key]))
            self.vars[key] = var

            scale = tk.Scale(
                self,
                from_=meta["min"],
                to=meta["max"],
                resolution=meta["step"],
                orient=tk.HORIZONTAL,
                variable=var,
                showvalue=False,
                length=300,
                command=lambda value, k=key, is_int=is_int: self._on_scale_change(k, value, is_int),
                bg=BG_COLOR,
                fg=FG_COLOR,
                highlightthickness=1,
                highlightbackground=CANVAS_BORDER,
                highlightcolor=CANVAS_BORDER,
                troughcolor=SCALE_TROUGH,
                activebackground=BUTTON_ACTIVE_BG,
                sliderrelief="flat",
            )
            scale.grid(row=row, column=1, sticky="ew", pady=5)

            value_label = tk.Label(
                self,
                text=self._format_value(key, var.get()),
                bg=BG_COLOR,
                fg=FG_COLOR,
                width=10,
                anchor="w",
                font=("Segoe UI", 12),
            )
            value_label.grid(row=row, column=2, sticky="w", padx=(16, 0))
            self.value_labels[key] = value_label

        button_frame = tk.Frame(self, bg=BG_COLOR)
        button_frame.grid(row=len(PARAMS) + 1, column=0, columnspan=3, pady=(16, 0), sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_columnconfigure(2, weight=1)

        button_kwargs = dict(
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_ACTIVE_FG,
            relief="flat",
            bd=0,
            highlightbackground=BUTTON_BORDER,
            font=("Segoe UI", 12),
            width=14,
        )

        reset_btn = tk.Button(
            button_frame,
            text="Reset defaults",
            command=self.reset_defaults,
            **button_kwargs,
        )
        reset_btn.grid(row=0, column=0, padx=(0, 12), sticky="ew")

        save_btn = tk.Button(
            button_frame,
            text="Save changes",
            command=self.save,
            **button_kwargs,
        )
        save_btn.grid(row=0, column=1, padx=12, sticky="ew")

        close_btn = tk.Button(
            button_frame,
            text="Close",
            command=self.close,
            **button_kwargs,
        )
        close_btn.grid(row=0, column=2, padx=(12, 0), sticky="ew")

        for key, var in self.vars.items():
            self._update_value_label(key, float(var.get()))

    def _on_scale_change(self, key: str, value: str, is_int: bool) -> None:
        numeric = float(value)
        if is_int:
            numeric = int(round(numeric))
            self.vars[key].set(numeric)
        self._update_value_label(key, numeric)

    def _format_value(self, key: str, value: float) -> str:
        if key in {"epochs", "batch_size", "step_size", "seed"}:
            return f"{int(round(value))}"
        if key in {"val_ratio", "test_ratio"}:
            return f"{value:.2f}"
        if key == "dropout":
            return f"{value:.2f}"
        if key == "learning_rate":
            return f"{value:.5f}"
        if key == "lr_gamma":
            return f"{value:.2f}"
        return f"{value:.3f}"

    def _update_value_label(self, key: str, value: float) -> None:
        label = self.value_labels.get(key)
        if label is not None:
            label.config(text=self._format_value(key, value))

    def _collect_values(self) -> dict:
        data: dict[str, float | int] = {}
        for meta in PARAMS:
            key = meta["key"]
            raw = float(self.vars[key].get())
            if meta["type"] == "int":
                data[key] = int(round(raw))
            else:
                data[key] = round(raw, 5)
        return data

    def save(self) -> bool:
        """Uloží hodnoty do config.json; neplatné neuloží a ohlásí. Vrací, zda se uložilo."""
        data = self._collect_values()
        try:
            validate_values(data)
        except ValueError as exc:
            messagebox.showerror("Invalid values", f"{exc}\n\nConfig was not saved.")
            return False
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CONFIG_PATH.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Save failed", str(exc))
            return False
        return True

    def _save_and_close(self) -> None:
        # Při neplatných hodnotách okno zůstane otevřené, aby šly opravit.
        if self.save():
            self.close()

    def reset_defaults(self) -> None:
        for meta in PARAMS:
            key = meta["key"]
            default_value = DEFAULTS[key]
            self.vars[key].set(default_value)
            self._update_value_label(key, float(default_value))

    def close(self) -> None:
        self.destroy()


def main() -> None:
    app = HyperparamEditor()
    app.mainloop()


if __name__ == "__main__":
    main()
