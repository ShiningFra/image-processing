import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk
import time

from image_loader import load_image, to_gray, to_pil
from algorithms import (
    histogram, contrast, filters, fourier,
    edges, thresholding, morphology,
    kmeans_segmentation, svd_compression
)

# ---------- THEME ----------
BG_MAIN = "#1e1e2e"
BG_SIDEBAR = "#25253a"
BG_SECTION = "#2f2f4f"
BTN_COLOR = "#4e8cff"
TXT_COLOR = "#ffffff"

IMG_SIZE = (420, 320)

class VisionStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vision Studio – Traitement d’Images")
        self.geometry("1300x760")
        self.configure(bg=BG_MAIN)

        self.original_img = None
        self.processed_img = None
        self.current_algo = None

        self.build_ui()

    def _on_canvas_configure(self, event):
        # Permet à la frame interne de prendre toute la largeur du canvas
        self.main_canvas.itemconfig(self.canvas_window, width=event.width)

    # ---------- UI ----------
    def build_ui(self):
        sidebar = tk.Frame(self, bg=BG_SIDEBAR, width=300)
        sidebar.pack(side="left", fill="y")

        tk.Label(
            sidebar, text="VISION STUDIO",
            bg=BG_SIDEBAR, fg=TXT_COLOR,
            font=("Segoe UI", 14, "bold")
        ).pack(pady=10)

        tk.Button(
            sidebar, text="📂 Charger une image",
            command=self.load_image,
            bg=BTN_COLOR, fg="white",
            relief="flat", pady=8
        ).pack(fill="x", padx=15, pady=10)

        # ---- Scrollable algorithms ----
        canvas = tk.Canvas(sidebar, bg=BG_SIDEBAR, highlightthickness=0)
        scroll = tk.Scrollbar(sidebar, command=canvas.yview)
        self.algo_frame = tk.Frame(canvas, bg=BG_SIDEBAR)

        self.algo_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.algo_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # ---- Algorithms ----
        self.add_section("Amélioration", [
            ("Histogramme", lambda: self.run("Histogramme",
                lambda: histogram.histogram_equalization(self.gray()))),
            ("Gamma", lambda: self.set_params(
                "Gamma", self.gamma_panel,
                lambda: contrast.gamma_correction(self.gray(), self.gamma.get())
            ))
        ])

        self.add_section("Filtrage spatial", [
            ("Moyenneur", lambda: self.set_params(
                "Moyenneur", self.kernel_panel,
                lambda: filters.mean_filter(self.gray(), self.kernel.get())
            )),
            ("Gaussien", lambda: self.set_params(
                "Gaussien", self.kernel_panel,
                lambda: filters.gaussian_filter(self.gray(), self.kernel.get())
            )),
            ("Médian", lambda: self.set_params(
                "Médian", self.kernel_panel,
                lambda: filters.median_filter(self.gray(), self.kernel.get())
            ))
        ])

        self.add_section("Analyse fréquentielle", [
            ("Fourier passe-bas", lambda: self.set_params(
                "Fourier", self.radius_panel,
                lambda: fourier.low_pass_filter(self.gray(), self.radius.get())
            ))
        ])

        self.add_section("Segmentation", [
            ("Otsu", lambda: self.run("Otsu",
                lambda: thresholding.otsu_threshold(self.gray()))),
            ("K-means", lambda: self.set_params(
                "K-means", self.kmeans_panel,
                lambda: kmeans_segmentation.kmeans_segment(
                    self.original_img, self.k_clusters.get()
                )
            ))
        ])

        self.add_section("Morphologie", [
            ("Érosion", lambda: self.set_params(
                "Érosion", self.kernel_panel,
                lambda: morphology.erosion(self.binary())
            )),
            ("Dilatation", lambda: self.set_params(
                "Dilatation", self.kernel_panel,
                lambda: morphology.dilation(self.binary())
            ))
        ])

        self.add_section("Compression", [
            ("SVD", lambda: self.set_params(
                "SVD", self.svd_panel,
                lambda: svd_compression.svd_compress(self.gray(), self.svd_k.get())
            ))
        ])

        # ---- Main area (with Scrollbar) ----
        main_container = tk.Frame(self, bg=BG_MAIN)
        main_container.pack(side="right", fill="both", expand=True)

        # Création du Canvas et de la Scrollbar
        self.main_canvas = tk.Canvas(main_container, bg=BG_MAIN, highlightthickness=0)
        main_scrollbar = tk.Scrollbar(main_container, orient="vertical", command=self.main_canvas.yview)
        
        # Frame qui contiendra réellement les images
        self.scrollable_body = tk.Frame(self.main_canvas, bg=BG_MAIN)

        # Liaison entre Canvas et Scrollbar
        self.main_canvas.configure(yscrollcommand=main_scrollbar.set)
        
        # Placement
        main_scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)
        
        # Insertion de la Frame dans le Canvas
        self.canvas_window = self.main_canvas.create_window((0, 0), window=self.scrollable_body, anchor="nw")

        # Configuration du redimensionnement
        self.scrollable_body.bind("<Configure>", lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))
        
        # Ajuster la largeur du contenu à celle du canvas
        self.main_canvas.bind('<Configure>', self._on_canvas_configure)

        # ---- Éléments déplacés dans la scrollable_body ----
        self.orig_canvas = tk.Label(self.scrollable_body, bg=BG_MAIN)
        self.proc_canvas = tk.Label(self.scrollable_body, bg=BG_MAIN)
        self.orig_canvas.pack(pady=10, padx=20)
        self.proc_canvas.pack(pady=10, padx=20)

        self.status = tk.Label(self.scrollable_body, bg=BG_MAIN, fg="orange")
        self.status.pack()

        # ---- Param panel (déplacé dans la zone scrollable) ----
        self.param_frame = tk.LabelFrame(
            self.scrollable_body, text="Paramètres",
            bg=BG_SECTION, fg="white"
        )
        self.param_frame.pack(fill="x", padx=30, pady=20)

        # Init variables
        self.gamma = tk.DoubleVar(value=1.5)
        self.kernel = tk.IntVar(value=3)
        self.radius = tk.IntVar(value=30)
        self.k_clusters = tk.IntVar(value=3)
        self.svd_k = tk.IntVar(value=50)

    # ---------- Sections ----------
    def add_section(self, title, buttons):
        frame = tk.LabelFrame(
            self.algo_frame, text=title,
            bg=BG_SECTION, fg="white"
        )
        frame.pack(fill="x", padx=10, pady=6)

        for txt, cmd in buttons:
            tk.Button(
                frame, text=txt, command=cmd,
                bg=BTN_COLOR, fg="white",
                relief="flat", pady=6
            ).pack(fill="x", pady=3)

    # ---------- Parameter Panels ----------
    def clear_params(self):
        for w in self.param_frame.winfo_children():
            w.destroy()

    def gamma_panel(self):
        tk.Label(self.param_frame, text="Gamma", bg=BG_SECTION, fg="white").pack()
        tk.Scale(self.param_frame, from_=0.1, to=3.0,
                 resolution=0.1, variable=self.gamma,
                 orient="horizontal", bg=BG_SECTION,
                 fg="white").pack(fill="x")

    def kernel_panel(self):
        tk.Label(self.param_frame, text="Taille noyau (impair)",
                 bg=BG_SECTION, fg="white").pack()
        tk.Spinbox(self.param_frame, from_=3, to=15,
                   increment=2, textvariable=self.kernel).pack()

    def radius_panel(self):
        tk.Label(self.param_frame, text="Rayon fréquentiel",
                 bg=BG_SECTION, fg="white").pack()
        tk.Scale(self.param_frame, from_=5, to=100,
                 variable=self.radius, orient="horizontal",
                 bg=BG_SECTION, fg="white").pack(fill="x")

    def kmeans_panel(self):
        tk.Label(self.param_frame, text="Nombre de clusters",
                 bg=BG_SECTION, fg="white").pack()
        tk.Spinbox(self.param_frame, from_=2, to=10,
                   textvariable=self.k_clusters).pack()

    def svd_panel(self):
        tk.Label(self.param_frame, text="Valeurs singulières k",
                 bg=BG_SECTION, fg="white").pack()
        tk.Scale(self.param_frame, from_=5, to=150,
                 variable=self.svd_k, orient="horizontal",
                 bg=BG_SECTION, fg="white").pack(fill="x")

    # ---------- Logic ----------
    def set_params(self, name, panel_func, algo_func):
        if self.original_img is None:
            messagebox.showerror("Erreur", "Charge une image d'abord")
            return

        self.clear_params()
        panel_func()

        tk.Button(
            self.param_frame, text="▶ Appliquer",
            bg="#00c853", fg="white",
            command=lambda: self.run(name, algo_func)
        ).pack(pady=8)

    def run(self, name, func):
        start = time.time()
        self.processed_img = func()
        self.show(self.processed_img, self.proc_canvas)
        self.status.config(text=f"{name} exécuté en {round(time.time()-start,3)} s")

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return

        self.original_img = load_image(path)
        self.show(self.original_img, self.orig_canvas)
        self.status.config(text="Image chargée")

    def show(self, img, canvas):
        pil = to_pil(img).resize(IMG_SIZE)
        tk_img = ImageTk.PhotoImage(pil)
        canvas.config(image=tk_img)
        canvas.image = tk_img

    def gray(self):
        return to_gray(self.original_img)

    def binary(self):
        return thresholding.otsu_threshold(self.gray())

if __name__ == "__main__":
    VisionStudio().mainloop()
