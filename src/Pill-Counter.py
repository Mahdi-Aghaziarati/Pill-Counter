import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
import csv
import json
import os
from datetime import datetime
import platform
import subprocess
import sys

# ----------------------------------------------------------------------
# Pill Counter Application
# ----------------------------------------------------------------------
class PillCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Pill Counter - Full Feature Set")
        self.root.geometry("1500x950")
        self.root.configure(bg='#2c3e50')

        # ---------- Camera & Threads ----------
        self.cap = None
        self.running = False
        self.video_thread = None
        self.camera_index = 0
        self.camera_names_dict = {}          # index -> friendly name
        self.target_fps = 30
        self.last_frame_time = 0

        # ---------- Resolution ----------
        self.resolutions = ["320x240", "640x480", "800x600", "1024x768", "1280x720", "1920x1080"]
        self.current_resolution = "640x480"

        # ---------- Detection Parameters ----------
        self.use_auto_threshold = True
        self.threshold_value = 50
        self.invert_threshold = False

        self.blur_kernel = 5                 # Gaussian (odd)
        self.morph_close_iter = 1            # Closing iterations (fill holes)
        self.morph_open_iter = 1             # Opening iterations (remove noise)
        self.circularity_threshold = 0.6     # 0..1, 1 = perfect circle
        self.min_area = 30                   # can go down to 1
        self.max_area = 3000
        self.use_watershed = True
        self.erosion_iterations = 1          # separate touching pills
        self.roi_percent = 80                # central ROI size (%)
        self.roi_shape = "rectangle"         # rectangle or circle

        # ---------- Advanced Options ----------
        self.use_color_filter = False         # filter by HSV
        self.hsv_lower = np.array([0, 0, 200])  # default: white-ish
        self.hsv_upper = np.array([180, 30, 255])
        self.use_background_subtractor = False
        self.bg_subtractor = None
        self.learning_rate = 0.01

        # ---------- Display Options ----------
        self.show_contours = True
        self.show_centroid_numbers = True
        self.show_fps = True
        self.show_processing_time = True

        # ---------- Logging ----------
        self.log_enabled = False
        self.log_file = None
        self.last_log_time = 0

        # ---------- Snapshot & Settings ----------
        self.last_annotated_frame = None
        self.settings_file = "pill_counter_settings.json"

        # ---------- Build GUI ----------
        self.create_widgets()
        self.enumerate_cameras()
        self.load_settings()

    # ------------------------------------------------------------------
    # Camera enumeration with friendly names
    # ------------------------------------------------------------------
    def enumerate_cameras(self):
        """Detect available cameras and try to get real device names."""
        self.camera_names_dict.clear()
        available = []

        # On Windows, use DirectShow to query names
        if platform.system() == "Windows":
            try:
                # Use pygrab or just try to open with CAP_DSHOW and retrieve backend name?
                # Simpler: use OpenCV's backend, but it doesn't give names.
                # Alternative: run `ffmpeg -list_devices true -f dshow -i dummy` (if ffmpeg installed)
                # For robustness, we use a manual method: try indices 0-9 and assign generic names.
                pass
            except:
                pass
        # For all OS, we try indices 0-9 and build a name list
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
            if cap.isOpened():
                # Try to get backend-specific name (not reliable)
                name = f"Camera {i}"
                # On Linux, you could read /sys/class/video4linux/video{i}/name
                if platform.system() == "Linux":
                    try:
                        with open(f"/sys/class/video4linux/video{i}/name", "r") as f:
                            name = f.read().strip()
                    except:
                        pass
                self.camera_names_dict[i] = name
                available.append(name)
                cap.release()
        if not available:
            available = ["No camera found (index 0)"]
            self.camera_names_dict[0] = "Default Camera"
        self.cam_combo['values'] = available
        self.cam_combo.current(0)
        self.camera_index = list(self.camera_names_dict.keys())[0]

    # ------------------------------------------------------------------
    # GUI Creation (full featured)
    # ------------------------------------------------------------------
    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ----- Left side: Video feeds (original + processed) -----
        left_panel = tk.Frame(main_frame, bg='black')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left_panel, text="Original Feed (with ROI and detections)", bg='black', fg='white', font=('Arial', 12)).pack()
        self.original_label = tk.Label(left_panel, bg='black')
        self.original_label.pack(fill=tk.BOTH, expand=True, pady=(0,10))

        tk.Label(left_panel, text="Processed Mask (after segmentation)", bg='black', fg='white', font=('Arial', 12)).pack()
        self.processed_label = tk.Label(left_panel, bg='black')
        self.processed_label.pack(fill=tk.BOTH, expand=True)

        # ----- Right side: Controls (scrolled) -----
        right_canvas = tk.Canvas(main_frame, bg='#34495e', width=400)
        scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=right_canvas.yview)
        self.scrollable_frame = tk.Frame(right_canvas, bg='#34495e')
        self.scrollable_frame.bind("<Configure>", lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.create_window((0,0), window=self.scrollable_frame, anchor="nw")
        right_canvas.configure(yscrollcommand=scrollbar.set)
        right_canvas.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel = self.scrollable_frame

        # Title
        tk.Label(right_panel, text="Pill Counter - Full Control", font=('Arial', 14, 'bold'),
                 bg='#34495e', fg='white').pack(pady=10)

        # ----- Camera Section -----
        cam_frame = tk.LabelFrame(right_panel, text="Camera", bg='#34495e', fg='white')
        cam_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(cam_frame, text="Select Camera:", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.cam_combo = ttk.Combobox(cam_frame, values=[], width=35)
        self.cam_combo.pack(padx=5, pady=2)
        self.cam_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        self.refresh_btn = tk.Button(cam_frame, text="Refresh Cameras", command=self.enumerate_cameras, bg='#1abc9c', fg='white')
        self.refresh_btn.pack(pady=2)

        # Resolution & FPS
        res_frame = tk.LabelFrame(right_panel, text="Resolution & FPS", bg='#34495e', fg='white')
        res_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(res_frame, text="Resolution:", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.res_combo = ttk.Combobox(res_frame, values=self.resolutions, width=35)
        self.res_combo.set(self.current_resolution)
        self.res_combo.pack(padx=5, pady=2)
        self.res_combo.bind("<<ComboboxSelected>>", self.on_resolution_changed)
        tk.Label(res_frame, text="Target FPS:", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.fps_slider = tk.Scale(res_frame, from_=5, to=60, orient=tk.HORIZONTAL, command=self.update_fps, bg='#34495e', fg='white')
        self.fps_slider.set(self.target_fps)
        self.fps_slider.pack(fill=tk.X, padx=10)
        self.fps_label = tk.Label(res_frame, text="Target: 30 FPS", bg='#34495e', fg='white')
        self.fps_label.pack()

        # Control buttons (Start/Stop/Snapshot)
        btn_frame = tk.Frame(right_panel, bg='#34495e')
        btn_frame.pack(pady=10)
        self.start_btn = tk.Button(btn_frame, text="▶ Start", command=self.start_video, bg='#2ecc71', fg='white', width=10)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop", command=self.stop_video, bg='#e74c3c', fg='white', width=10, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.snapshot_btn = tk.Button(btn_frame, text="📸 Snapshot", command=self.save_snapshot, bg='#f39c12', fg='white', width=10)
        self.snapshot_btn.pack(side=tk.LEFT, padx=5)

        # Real-time stats
        stats_frame = tk.LabelFrame(right_panel, text="Real-time Stats", bg='#34495e', fg='white')
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        self.count_label = tk.Label(stats_frame, text="Count: 0", bg='#34495e', fg='#f1c40f', font=('Arial', 24, 'bold'))
        self.count_label.pack(pady=5)
        self.actual_fps_label = tk.Label(stats_frame, text="Actual FPS: --", bg='#34495e', fg='#1abc9c')
        self.actual_fps_label.pack()
        self.proc_time_label = tk.Label(stats_frame, text="Proc. time: -- ms", bg='#34495e', fg='#1abc9c')
        self.proc_time_label.pack()

        # Logging
        log_frame = tk.LabelFrame(right_panel, text="Logging", bg='#34495e', fg='white')
        log_frame.pack(fill=tk.X, padx=10, pady=5)
        self.log_var = tk.BooleanVar(value=False)
        log_cb = tk.Checkbutton(log_frame, text="Enable CSV logging (count + timestamp)", variable=self.log_var,
                                command=self.toggle_logging, bg='#34495e', fg='white', selectcolor='#34495e')
        log_cb.pack(anchor='w', padx=5)
        self.log_file_label = tk.Label(log_frame, text="No log file", bg='#34495e', fg='#bbb')
        self.log_file_label.pack(anchor='w', padx=5)

        # ----- Detection Parameters (all sliders) -----
        det_frame = tk.LabelFrame(right_panel, text="Detection Parameters", bg='#34495e', fg='white')
        det_frame.pack(fill=tk.X, padx=10, pady=10)

        # Auto / Manual threshold
        self.auto_thresh_var = tk.BooleanVar(value=self.use_auto_threshold)
        auto_cb = tk.Checkbutton(det_frame, text="Auto Threshold (Otsu)", variable=self.auto_thresh_var,
                                 command=self.toggle_auto_thresh, bg='#34495e', fg='white', selectcolor='#34495e')
        auto_cb.pack(anchor='w', padx=5)
        self.thresh_slider = tk.Scale(det_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                      label="Manual Threshold", command=self.update_thresh, bg='#34495e', fg='white')
        self.thresh_slider.set(self.threshold_value)
        self.thresh_slider.pack(fill=tk.X, padx=10)
        # Invert
        self.invert_var = tk.BooleanVar(value=self.invert_threshold)
        inv_cb = tk.Checkbutton(det_frame, text="Invert Threshold (dark pills on bright bg)", variable=self.invert_var,
                                command=self.toggle_invert, bg='#34495e', fg='white', selectcolor='#34495e')
        inv_cb.pack(anchor='w', padx=5)

        # Blur
        tk.Label(det_frame, text="Gaussian Blur Kernel (odd):", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.blur_slider = tk.Scale(det_frame, from_=1, to=21, resolution=2, orient=tk.HORIZONTAL,
                                    command=self.update_blur, bg='#34495e', fg='white')
        self.blur_slider.set(self.blur_kernel)
        self.blur_slider.pack(fill=tk.X, padx=10)

        # Morphology: Closing iterations
        tk.Label(det_frame, text="Closing Iterations (fill holes):", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.morph_close_slider = tk.Scale(det_frame, from_=0, to=5, orient=tk.HORIZONTAL,
                                           command=self.update_morph_close, bg='#34495e', fg='white')
        self.morph_close_slider.set(self.morph_close_iter)
        self.morph_close_slider.pack(fill=tk.X, padx=10)

        # Morphology: Opening iterations
        tk.Label(det_frame, text="Opening Iterations (remove noise):", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.morph_open_slider = tk.Scale(det_frame, from_=0, to=5, orient=tk.HORIZONTAL,
                                          command=self.update_morph_open, bg='#34495e', fg='white')
        self.morph_open_slider.set(self.morph_open_iter)
        self.morph_open_slider.pack(fill=tk.X, padx=10)

        # Circularity
        tk.Label(det_frame, text="Circularity (0 = any shape, 1 = perfect circle):", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.circ_slider = tk.Scale(det_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                                    command=self.update_circularity, bg='#34495e', fg='white')
        self.circ_slider.set(self.circularity_threshold)
        self.circ_slider.pack(fill=tk.X, padx=10)

        # Min/Max area (extended range)
        tk.Label(det_frame, text="Min Pill Area (pixels) [1-10000]:", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.min_area_slider = tk.Scale(det_frame, from_=1, to=10000, orient=tk.HORIZONTAL,
                                        command=self.update_min_area, bg='#34495e', fg='white')
        self.min_area_slider.set(self.min_area)
        self.min_area_slider.pack(fill=tk.X, padx=10)

        tk.Label(det_frame, text="Max Pill Area (pixels) [1-10000]:", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.max_area_slider = tk.Scale(det_frame, from_=1, to=10000, orient=tk.HORIZONTAL,
                                        command=self.update_max_area, bg='#34495e', fg='white')
        self.max_area_slider.set(self.max_area)
        self.max_area_slider.pack(fill=tk.X, padx=10)

        # Erosion & Watershed
        tk.Label(det_frame, text="Erosion Iterations (separate touching):", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.erosion_slider = tk.Scale(det_frame, from_=0, to=5, orient=tk.HORIZONTAL,
                                       command=self.update_erosion, bg='#34495e', fg='white')
        self.erosion_slider.set(self.erosion_iterations)
        self.erosion_slider.pack(fill=tk.X, padx=10)

        self.watershed_var = tk.BooleanVar(value=self.use_watershed)
        ws_cb = tk.Checkbutton(det_frame, text="Use Watershed Separation", variable=self.watershed_var,
                               command=self.toggle_watershed, bg='#34495e', fg='white', selectcolor='#34495e')
        ws_cb.pack(anchor='w', padx=5)

        # ROI
        tk.Label(det_frame, text="ROI central region (% of frame):", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.roi_slider = tk.Scale(det_frame, from_=10, to=100, orient=tk.HORIZONTAL,
                                   command=self.update_roi, bg='#34495e', fg='white')
        self.roi_slider.set(self.roi_percent)
        self.roi_slider.pack(fill=tk.X, padx=10)

        tk.Label(det_frame, text="ROI Shape:", bg='#34495e', fg='white').pack(anchor='w', padx=5)
        self.roi_shape_var = tk.StringVar(value=self.roi_shape)
        shape_menu = ttk.Combobox(det_frame, textvariable=self.roi_shape_var, values=["rectangle", "circle"], width=15)
        shape_menu.pack(padx=5, pady=2)
        shape_menu.bind("<<ComboboxSelected>>", lambda e: setattr(self, 'roi_shape', self.roi_shape_var.get()))

        # ----- Advanced Filters -----
        adv_frame = tk.LabelFrame(right_panel, text="Advanced Filters", bg='#34495e', fg='white')
        adv_frame.pack(fill=tk.X, padx=10, pady=10)

        self.color_filter_var = tk.BooleanVar(value=self.use_color_filter)
        color_cb = tk.Checkbutton(adv_frame, text="Enable HSV Color Filter", variable=self.color_filter_var,
                                  command=self.toggle_color_filter, bg='#34495e', fg='white', selectcolor='#34495e')
        color_cb.pack(anchor='w', padx=5)

        # HSV sliders (simple GUI, could be improved)
        hsv_frame = tk.Frame(adv_frame, bg='#34495e')
        hsv_frame.pack(fill=tk.X, padx=5)
        tk.Label(hsv_frame, text="Hue min/max (0-180):", bg='#34495e', fg='white').pack(anchor='w')
        self.hue_min = tk.Scale(hsv_frame, from_=0, to=180, orient=tk.HORIZONTAL, bg='#34495e', fg='white')
        self.hue_min.set(0)
        self.hue_min.pack(fill=tk.X)
        self.hue_max = tk.Scale(hsv_frame, from_=0, to=180, orient=tk.HORIZONTAL, bg='#34495e', fg='white')
        self.hue_max.set(180)
        self.hue_max.pack(fill=tk.X)
        tk.Label(hsv_frame, text="Sat min/max (0-255):", bg='#34495e', fg='white').pack(anchor='w')
        self.sat_min = tk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, bg='#34495e', fg='white')
        self.sat_min.set(0)
        self.sat_min.pack(fill=tk.X)
        self.sat_max = tk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, bg='#34495e', fg='white')
        self.sat_max.set(255)
        self.sat_max.pack(fill=tk.X)
        tk.Label(hsv_frame, text="Val min/max (0-255):", bg='#34495e', fg='white').pack(anchor='w')
        self.val_min = tk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, bg='#34495e', fg='white')
        self.val_min.set(200)
        self.val_min.pack(fill=tk.X)
        self.val_max = tk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, bg='#34495e', fg='white')
        self.val_max.set(255)
        self.val_max.pack(fill=tk.X)

        # Background subtractor
        self.bgsub_var = tk.BooleanVar(value=self.use_background_subtractor)
        bg_cb = tk.Checkbutton(adv_frame, text="Enable Background Subtractor (MOG2)", variable=self.bgsub_var,
                               command=self.toggle_bgsub, bg='#34495e', fg='white', selectcolor='#34495e')
        bg_cb.pack(anchor='w', padx=5)

        # ----- Display Options -----
        disp_frame = tk.LabelFrame(right_panel, text="Display Options", bg='#34495e', fg='white')
        disp_frame.pack(fill=tk.X, padx=10, pady=10)

        self.show_contours_var = tk.BooleanVar(value=self.show_contours)
        cont_cb = tk.Checkbutton(disp_frame, text="Draw Contours", variable=self.show_contours_var,
                                 command=lambda: setattr(self, 'show_contours', self.show_contours_var.get()),
                                 bg='#34495e', fg='white', selectcolor='#34495e')
        cont_cb.pack(anchor='w', padx=5)

        self.show_numbers_var = tk.BooleanVar(value=self.show_centroid_numbers)
        num_cb = tk.Checkbutton(disp_frame, text="Show Centroid Numbers", variable=self.show_numbers_var,
                                command=lambda: setattr(self, 'show_centroid_numbers', self.show_numbers_var.get()),
                                bg='#34495e', fg='white', selectcolor='#34495e')
        num_cb.pack(anchor='w', padx=5)

        # ----- Save/Load & Reset -----
        settings_frame = tk.Frame(right_panel, bg='#34495e')
        settings_frame.pack(pady=10)
        tk.Button(settings_frame, text="Save Settings", command=self.save_settings, bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(settings_frame, text="Load Settings", command=self.load_settings, bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(settings_frame, text="Reset to Defaults", command=self.reset_defaults, bg='#e67e22', fg='white').pack(side=tk.LEFT, padx=5)

        # Help text
        help_text = """
        Tips for best results:
        - Black background + bright pills → use Otsu auto threshold.
        - For touching pills: enable Watershed, set Erosion=1-2.
        - Very small pills: set Min Area=1-10.
        - Use color filter if pills have specific hue.
        - ROI focuses on central area, ignoring edges.
        """
        help_label = tk.Label(right_panel, text=help_text, bg='#34495e', fg='#ecf0f1', justify=tk.LEFT, font=('Arial', 9))
        help_label.pack(pady=10, fill=tk.X, padx=10)

    # ------------------------------------------------------------------
    # Parameter update callbacks
    # ------------------------------------------------------------------
    def toggle_auto_thresh(self):
        self.use_auto_threshold = self.auto_thresh_var.get()
        self.thresh_slider.config(state=tk.DISABLED if self.use_auto_threshold else tk.NORMAL)

    def update_thresh(self, val):
        self.threshold_value = int(val)

    def toggle_invert(self):
        self.invert_threshold = self.invert_var.get()

    def update_blur(self, val):
        val = int(val)
        if val % 2 == 0:
            val += 1
        self.blur_kernel = max(1, val)

    def update_morph_close(self, val):
        self.morph_close_iter = int(val)

    def update_morph_open(self, val):
        self.morph_open_iter = int(val)

    def update_circularity(self, val):
        self.circularity_threshold = float(val)

    def update_min_area(self, val):
        self.min_area = int(val)

    def update_max_area(self, val):
        self.max_area = int(val)

    def update_erosion(self, val):
        self.erosion_iterations = int(val)

    def toggle_watershed(self):
        self.use_watershed = self.watershed_var.get()

    def update_roi(self, val):
        self.roi_percent = int(val)

    def update_fps(self, val):
        self.target_fps = int(val)
        self.fps_label.config(text=f"Target: {self.target_fps} FPS")

    def toggle_color_filter(self):
        self.use_color_filter = self.color_filter_var.get()
        if self.use_color_filter:
            self.hsv_lower = np.array([self.hue_min.get(), self.sat_min.get(), self.val_min.get()])
            self.hsv_upper = np.array([self.hue_max.get(), self.sat_max.get(), self.val_max.get()])

    def toggle_bgsub(self):
        self.use_background_subtractor = self.bgsub_var.get()
        if self.use_background_subtractor:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        else:
            self.bg_subtractor = None

    # ------------------------------------------------------------------
    # Camera and video control
    # ------------------------------------------------------------------
    def on_camera_selected(self, event):
        selected = self.cam_combo.get()
        for idx, name in self.camera_names_dict.items():
            if name == selected:
                self.camera_index = idx
                break
        if self.running:
            was_running = True
            self.stop_video()
            self.start_video()
        else:
            pass

    def on_resolution_changed(self, event):
        self.current_resolution = self.res_combo.get()
        if self.running:
            was_running = True
            self.stop_video()
            self.start_video()

    def set_camera_resolution(self, cap):
        w, h = map(int, self.current_resolution.split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def start_video(self):
        if self.running:
            return
        cap_backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, cap_backend)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera!")
            return
        self.set_camera_resolution(self.cap)
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.log_enabled and self.log_file:
            self.log_file.close()
            self.log_file = None

    def save_snapshot(self):
        if self.last_annotated_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.png"
            cv2.imwrite(filename, self.last_annotated_frame)
            messagebox.showinfo("Snapshot", f"Saved as {filename}")
        else:
            messagebox.showwarning("No frame", "No frame available yet.")

    def toggle_logging(self):
        self.log_enabled = self.log_var.get()
        if self.log_enabled:
            if not self.log_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pill_log_{timestamp}.csv"
                self.log_file = open(filename, 'w', newline='')
                writer = csv.writer(self.log_file)
                writer.writerow(["Timestamp", "PillCount"])
                self.log_file_label.config(text=f"Logging to {filename}")
            else:
                self.log_file_label.config(text=f"Logging to {self.log_file.name}")
        else:
            if self.log_file:
                self.log_file.close()
                self.log_file = None
                self.log_file_label.config(text="Logging stopped")

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------
    def video_loop(self):
        frame_time = 1.0 / max(self.target_fps, 1)
        fps_counter = 0
        fps_timer = time.time()
        log_timer = time.time()

        while self.running and self.cap:
            loop_start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame
            binary_mask, pill_count, annotated_frame, proc_time = self.process_frame(frame)
            self.last_annotated_frame = annotated_frame

            # Update UI
            self.root.after(0, lambda c=pill_count: self.count_label.config(text=f"Count: {c}"))
            self.root.after(0, lambda pt=proc_time: self.proc_time_label.config(text=f"Proc. time: {pt:.1f} ms"))

            # Display original
            orig_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            orig_img = Image.fromarray(orig_rgb)
            orig_img = self.resize_to_fit(orig_img, 640)
            orig_tk = ImageTk.PhotoImage(orig_img)
            self.root.after(0, lambda: self.original_label.config(image=orig_tk))
            self.root.after(0, lambda: setattr(self.original_label, 'image', orig_tk))

            # Display processed mask
            mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            proc_rgb = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
            proc_img = Image.fromarray(proc_rgb)
            proc_img = self.resize_to_fit(proc_img, 640)
            proc_tk = ImageTk.PhotoImage(proc_img)
            self.root.after(0, lambda: self.processed_label.config(image=proc_tk))
            self.root.after(0, lambda: setattr(self.processed_label, 'image', proc_tk))

            # FPS calculation
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                actual_fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()
                if self.show_fps:
                    self.root.after(0, lambda: self.actual_fps_label.config(text=f"Actual FPS: {actual_fps}"))
                else:
                    self.root.after(0, lambda: self.actual_fps_label.config(text=""))

            # Logging
            if self.log_enabled and time.time() - log_timer >= 1.0:
                log_timer = time.time()
                if self.log_file:
                    writer = csv.writer(self.log_file)
                    writer.writerow([datetime.now().isoformat(), pill_count])
                    self.log_file.flush()

            # Throttle
            elapsed = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self.cap:
            self.cap.release()

    def resize_to_fit(self, pil_img, max_width):
        w, h = pil_img.size
        if w > max_width:
            ratio = max_width / w
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return pil_img

    # ------------------------------------------------------------------
    # Core image processing (with all features)
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        start = time.time()
        h, w = frame.shape[:2]

        # 1. Optional: Background subtraction
        if self.use_background_subtractor and self.bg_subtractor:
            fgmask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
            # Use fgmask as base, but we will combine with threshold later
            gray = fgmask
        else:
            # 2. Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. Optional: Color filtering (HSV)
        if self.use_color_filter:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.hsv_lower = np.array([self.hue_min.get(), self.sat_min.get(), self.val_min.get()])
            self.hsv_upper = np.array([self.hue_max.get(), self.sat_max.get(), self.val_max.get()])
            color_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            # Combine with gray threshold later

        # 4. Blur and threshold
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        if self.use_auto_threshold:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)

        if self.invert_threshold:
            binary = cv2.bitwise_not(binary)

        # Combine with color mask if needed
        if self.use_color_filter:
            binary = cv2.bitwise_and(binary, color_mask)

        # 5. Morphology: Closing (fill holes), Opening (remove noise)
        kernel = np.ones((3,3), np.uint8)
        if self.morph_close_iter > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iter)
        if self.morph_open_iter > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=self.morph_open_iter)

        # 6. ROI (region of interest)
        roi_w = int(w * self.roi_percent / 100)
        roi_h = int(h * self.roi_percent / 100)
        start_x = (w - roi_w) // 2
        start_y = (h - roi_h) // 2
        end_x = start_x + roi_w
        end_y = start_y + roi_h

        roi_mask_full = np.zeros_like(binary)
        if self.roi_shape == "rectangle":
            roi_mask_full[start_y:end_y, start_x:end_x] = 255
        else:  # circle
            center = (w//2, h//2)
            radius = min(roi_w, roi_h) // 2
            cv2.circle(roi_mask_full, center, radius, 255, -1)
        binary = cv2.bitwise_and(binary, roi_mask_full)

        # 7. Erosion (separate touching pills)
        eroded = cv2.erode(binary, kernel, iterations=self.erosion_iterations)

        # 8. Watershed (advanced separation)
        if self.use_watershed:
            dist = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(eroded, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(frame, markers)
            final_mask = np.zeros_like(eroded)
            for label in range(2, np.max(markers) + 1):
                final_mask[markers == label] = 255
        else:
            final_mask = eroded

        # 9. Find contours and count pills
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = frame.copy()
        pill_count = 0

        # Draw ROI border
        if self.roi_shape == "rectangle":
            cv2.rectangle(annotated, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        else:
            center = (w//2, h//2)
            radius = min(roi_w, roi_h) // 2
            cv2.circle(annotated, center, radius, (255, 0, 0), 2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.circularity_threshold:
                continue
            pill_count += 1
            if self.show_contours:
                cv2.drawContours(annotated, [cnt], -1, (0, 255, 255), 2)
            if self.show_centroid_numbers:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(annotated, str(pill_count), (cx-8, cy-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Display count on frame
        cv2.putText(annotated, f"Pills: {pill_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        proc_time = (time.time() - start) * 1000
        return final_mask, pill_count, annotated, proc_time

    # ------------------------------------------------------------------
    # Settings save/load / reset
    # ------------------------------------------------------------------
    def save_settings(self):
        settings = {
            'use_auto_threshold': self.use_auto_threshold,
            'threshold_value': self.threshold_value,
            'invert_threshold': self.invert_threshold,
            'blur_kernel': self.blur_kernel,
            'morph_close_iter': self.morph_close_iter,
            'morph_open_iter': self.morph_open_iter,
            'circularity_threshold': self.circularity_threshold,
            'min_area': self.min_area,
            'max_area': self.max_area,
            'use_watershed': self.use_watershed,
            'erosion_iterations': self.erosion_iterations,
            'roi_percent': self.roi_percent,
            'roi_shape': self.roi_shape,
            'use_color_filter': self.use_color_filter,
            'use_background_subtractor': self.use_background_subtractor,
            'target_fps': self.target_fps,
            'current_resolution': self.current_resolution
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=4)
        messagebox.showinfo("Settings", f"Saved to {self.settings_file}")

    def load_settings(self):
        if not os.path.exists(self.settings_file):
            return
        with open(self.settings_file, 'r') as f:
            settings = json.load(f)
        self.use_auto_threshold = settings.get('use_auto_threshold', True)
        self.threshold_value = settings.get('threshold_value', 50)
        self.invert_threshold = settings.get('invert_threshold', False)
        self.blur_kernel = settings.get('blur_kernel', 5)
        self.morph_close_iter = settings.get('morph_close_iter', 1)
        self.morph_open_iter = settings.get('morph_open_iter', 1)
        self.circularity_threshold = settings.get('circularity_threshold', 0.6)
        self.min_area = settings.get('min_area', 30)
        self.max_area = settings.get('max_area', 3000)
        self.use_watershed = settings.get('use_watershed', True)
        self.erosion_iterations = settings.get('erosion_iterations', 1)
        self.roi_percent = settings.get('roi_percent', 80)
        self.roi_shape = settings.get('roi_shape', "rectangle")
        self.use_color_filter = settings.get('use_color_filter', False)
        self.use_background_subtractor = settings.get('use_background_subtractor', False)
        self.target_fps = settings.get('target_fps', 30)
        self.current_resolution = settings.get('current_resolution', "640x480")
        # Update GUI widgets
        self.auto_thresh_var.set(self.use_auto_threshold)
        self.thresh_slider.set(self.threshold_value)
        self.invert_var.set(self.invert_threshold)
        self.blur_slider.set(self.blur_kernel)
        self.morph_close_slider.set(self.morph_close_iter)
        self.morph_open_slider.set(self.morph_open_iter)
        self.circ_slider.set(self.circularity_threshold)
        self.min_area_slider.set(self.min_area)
        self.max_area_slider.set(self.max_area)
        self.watershed_var.set(self.use_watershed)
        self.erosion_slider.set(self.erosion_iterations)
        self.roi_slider.set(self.roi_percent)
        self.roi_shape_var.set(self.roi_shape)
        self.color_filter_var.set(self.use_color_filter)
        self.bgsub_var.set(self.use_background_subtractor)
        self.fps_slider.set(self.target_fps)
        self.res_combo.set(self.current_resolution)
        self.toggle_auto_thresh()
        self.toggle_color_filter()
        self.toggle_bgsub()
        messagebox.showinfo("Settings", "Loaded settings from file")

    def reset_defaults(self):
        self.use_auto_threshold = True
        self.threshold_value = 50
        self.invert_threshold = False
        self.blur_kernel = 5
        self.morph_close_iter = 1
        self.morph_open_iter = 1
        self.circularity_threshold = 0.6
        self.min_area = 30
        self.max_area = 3000
        self.use_watershed = True
        self.erosion_iterations = 1
        self.roi_percent = 80
        self.roi_shape = "rectangle"
        self.use_color_filter = False
        self.use_background_subtractor = False
        self.target_fps = 30
        self.current_resolution = "640x480"
        # Update UI
        self.auto_thresh_var.set(True)
        self.thresh_slider.set(50)
        self.invert_var.set(False)
        self.blur_slider.set(5)
        self.morph_close_slider.set(1)
        self.morph_open_slider.set(1)
        self.circ_slider.set(0.6)
        self.min_area_slider.set(30)
        self.max_area_slider.set(3000)
        self.watershed_var.set(True)
        self.erosion_slider.set(1)
        self.roi_slider.set(80)
        self.roi_shape_var.set("rectangle")
        self.color_filter_var.set(False)
        self.bgsub_var.set(False)
        self.fps_slider.set(30)
        self.res_combo.set("640x480")
        self.toggle_auto_thresh()
        self.toggle_color_filter()
        self.toggle_bgsub()
        messagebox.showinfo("Reset", "All parameters reset to defaults")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PillCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_video(), root.destroy()))
    root.mainloop()
