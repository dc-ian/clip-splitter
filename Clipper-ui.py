#!/usr/bin/env python3
"""
clipper_ui.py - Phase 4: desktop window for clipper.py

    pip install PySide6
    python clipper_ui.py

Keep clipper.py in the same folder. FFmpeg must be on your PATH.
Optional: pip install faster-whisper   (sentence-end cuts)
Optional: pip install yt-dlp           (paste a YouTube link instead of opening a file)

Flow:  drop a video (or paste a YouTube link) -> Analyze -> review/fix the clip list -> Export
  * clips that ended on a hard cut (no pause/scene/sentence found) are orange: check those first
  * click a clip to preview it; trim, merge, delete or re-export single clips
  * Export only re-encodes what changed (finished clips are kept)
  * Fast / Precise toggle (Fast = instant but keyframe-accurate only)
  * Captions (burned / .srt), 9:16 reframing and "best N clips only" in the Export options box
  * Ranking video: after Export, join the best N clips as a #N ... #1 countdown with rank badges
  * Cancel works at every stage, including Analyze

Package later with:  pyinstaller --noconsole --onefile clipper_ui.py
"""
import contextlib
import copy
import sys
import threading
from pathlib import Path

# Install first:  python -m pip install PySide6
from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QSpinBox, QSplitter,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import clipper

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".ts", ".wmv"}
SIGNAL_LABEL = {
    "sentence": "Sentence end", "silence": "Pause", "scene": "Scene change", "dip": "Quiet moment",
    "word": "Word gap", "hard": "Hard cut (check)", "end": "End of video", "fixed": "Fixed length",
    "manual": "Edited by you",
}
COLUMNS = ["#", "Start", "End", "Length", "Cut at", "Status", "Text"]
FAST_WARNING = ("Fast: near-instant, but cuts snap to keyframes, so a clip can start or end "
                "seconds away from what you see in the list, and there are no fades. "
                "Burned captions and 9:16 reframing need Precise mode (.srt files work in both).")


def fmt(sec):
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


class _LineWriter:
    """File-like object that turns print() output into one Qt signal per line."""

    def __init__(self, emit):
        self.emit, self.buf = emit, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.emit(line)
        return len(s)

    def flush(self):
        pass


class Worker(QThread):
    """Runs clipper.run_job off the GUI thread so the window never freezes."""
    log = Signal(str)
    event = Signal(object)
    failed = Signal(str)
    finished_ok = Signal()

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cancel_event = threading.Event()

    def run(self):
        try:
            with contextlib.redirect_stdout(_LineWriter(self.log.emit)):
                clipper.run_job(self.args, progress=self.event.emit, cancel=self.cancel_event)
            self.finished_ok.emit()
        except clipper.ClipperError as e:
            self.failed.emit(str(e))
        except Exception as e:  # never let a thread die silently
            self.failed.emit(f"{type(e).__name__}: {e}")


class DownloadWorker(QThread):
    """Downloads one online video off the GUI thread. Stop it with cancel_event."""
    log = Signal(str)
    progress = Signal(int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.cancel_event = threading.Event()

    def _on_progress(self, e):
        self.progress.emit(int(e.get("pct", 0)), e.get("text", ""))

    def run(self):
        try:
            with contextlib.redirect_stdout(_LineWriter(self.log.emit)):
                path = clipper.download_video(self.url, progress=self._on_progress, cancel=self.cancel_event)
            self.done.emit(path)
        except clipper.ClipperError as e:
            self.failed.emit(str(e))
        except Exception as e:  # never let a thread die silently
            self.failed.emit(f"{type(e).__name__}: {e}")


class RankWorker(QThread):
    """Builds the countdown (ranking) video from the exported clips, off the GUI thread."""
    log = Signal(str)
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, out_dir, top, title):
        super().__init__()
        self.out_dir, self.top, self.title = out_dir, top, title
        self.cancel_event = threading.Event()

    def _on_event(self, e):
        if e.get("stage") in ("rank_start", "rank_clip"):
            self.progress.emit(int(e.get("done", 0)), int(e.get("total", 1)))

    def run(self):
        try:
            with contextlib.redirect_stdout(_LineWriter(self.log.emit)):
                path = clipper.run_ranking(self.out_dir, self.top, self.title or None,
                                           progress=self._on_event, cancel=self.cancel_event)
            self.done.emit(path)
        except clipper.ClipperError as e:
            self.failed.emit(str(e))
        except Exception as e:  # never let a thread die silently
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clip Splitter")
        self.resize(1300, 800)
        self.setAcceptDrops(True)
        self.src = None
        self.out_dir = None
        self.manifest = None
        self.worker = None
        self.exporting = False
        self.preview_end = None
        self._quiet = 0     # > 0 while the code (not the user) changes the table selection: no autoplay
        self._build_ui()
        self._update_buttons()

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        split = QSplitter(Qt.Horizontal)
        self.setCentralWidget(split)

        # ---- left column
        left = QWidget()
        lv = QVBoxLayout(left)

        top = QHBoxLayout()
        self.file_label = QLabel("Drop a video here, paste a link below, or press Open")
        self.file_label.setStyleSheet("border: 2px dashed #888; padding: 14px;")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.btn_open = QPushButton("Open video...")
        self.btn_open.clicked.connect(self.on_open)
        top.addWidget(self.file_label, 1)
        top.addWidget(self.btn_open)
        lv.addLayout(top)

        dl = QHBoxLayout()
        self.ed_url = QLineEdit()
        self.ed_url.setPlaceholderText("Or paste a YouTube link here and press Enter")
        self.ed_url.setClearButtonEnabled(True)
        self.ed_url.returnPressed.connect(self.on_download)
        self.btn_dl = QPushButton("Download")
        self.btn_dl.clicked.connect(self.on_download)
        if not clipper.ytdlp_available():
            self.ed_url.setPlaceholderText("Paste a link here (needs yt-dlp: pip install yt-dlp, then restart)")
        dl.addWidget(self.ed_url, 1)
        dl.addWidget(self.btn_dl)
        lv.addLayout(dl)

        self.settings = QGroupBox("Settings")
        g = QGridLayout(self.settings)
        self.sp_min = QDoubleSpinBox(); self.sp_min.setRange(3, 120); self.sp_min.setValue(15); self.sp_min.setSuffix(" s")
        self.sp_max = QDoubleSpinBox(); self.sp_max.setRange(5, 180); self.sp_max.setValue(30); self.sp_max.setSuffix(" s")
        g.addWidget(QLabel("Shortest clip"), 0, 0); g.addWidget(self.sp_min, 0, 1)
        g.addWidget(QLabel("Longest clip"), 0, 2); g.addWidget(self.sp_max, 0, 3)
        self.chk_tr = QCheckBox("Use speech (Whisper) to cut at sentence ends")
        self.chk_tr.setChecked(clipper.whisper_available())
        if not clipper.whisper_available():
            self.chk_tr.setEnabled(False)
            self.chk_tr.setToolTip("Install with: pip install faster-whisper")
        self.cb_model = QComboBox(); self.cb_model.addItems(["tiny", "base", "small", "medium"]); self.cb_model.setCurrentText("base")
        self.ed_lang = QLineEdit(); self.ed_lang.setPlaceholderText("language (auto)"); self.ed_lang.setMaximumWidth(120)
        g.addWidget(self.chk_tr, 1, 0, 1, 2); g.addWidget(self.cb_model, 1, 2); g.addWidget(self.ed_lang, 1, 3)
        self.out_label = QLabel("Output folder: (next to the video)")
        self.btn_out = QPushButton("Choose...")
        self.btn_out.clicked.connect(self.on_choose_out)
        g.addWidget(self.out_label, 2, 0, 1, 3); g.addWidget(self.btn_out, 2, 3)
        lv.addWidget(self.settings)

        mode = QGroupBox("Export options")
        self.export_opts = mode
        mv = QVBoxLayout(mode)
        row = QHBoxLayout()
        self.rb_precise = QRadioButton("Precise (default, frame-accurate)"); self.rb_precise.setChecked(True)
        self.rb_fast = QRadioButton("Fast (keyframe cuts)")
        self.rb_fast.toggled.connect(self._on_mode_toggled)
        row.addWidget(self.rb_precise); row.addWidget(self.rb_fast)
        mv.addLayout(row)
        self.fast_warn = QLabel(FAST_WARNING)
        self.fast_warn.setWordWrap(True)
        self.fast_warn.setStyleSheet("color: #c0392b;")
        self.fast_warn.hide()
        mv.addWidget(self.fast_warn)
        og = QGridLayout()
        self.cb_caps = QComboBox()
        for label, val in (("Off", "off"), ("Burned into the video", "burn"),
                           ("Sidecar .srt files", "sidecar"), ("Both", "both")):
            self.cb_caps.addItem(label, val)
        self.cb_caps.setToolTip("Needs a transcript (pip install faster-whisper). "
                                "If the analysis skipped it, it is made when you export.")
        self.cb_vert = QComboBox()
        for label, val in (("Off (keep original shape)", "off"), ("Center crop", "center"),
                           ("Auto-pan to the speaker", "follow"), ("Blurred background", "blur")):
            self.cb_vert.addItem(label, val)
        self.sp_top = QSpinBox()
        self.sp_top.setRange(0, 9999)
        self.sp_top.setSpecialValueText("All clips")
        self.sp_top.setSuffix(" best clips")
        self.sp_top.setToolTip("Export only the highest-scoring clips. The rest are marked 'Not in best' "
                               "and can be brought back with Restore or by choosing All clips.")
        og.addWidget(QLabel("Captions"), 0, 0); og.addWidget(self.cb_caps, 0, 1)
        og.addWidget(QLabel("9:16 vertical"), 1, 0); og.addWidget(self.cb_vert, 1, 1)
        og.addWidget(QLabel("Export"), 2, 0); og.addWidget(self.sp_top, 2, 1)
        mv.addLayout(og)
        lv.addWidget(mode)

        self.rank_box = QGroupBox("Ranking video (countdown template)")
        self.rank_box.setToolTip("Picks the best exported clips by highlight score and joins them as a countdown, "
                                 "#N down to #1, with a rank badge, a title and a tracker on every clip. "
                                 "Export first.")
        rl = QHBoxLayout(self.rank_box)
        self.sp_rank = QSpinBox()
        self.sp_rank.setRange(1, 20); self.sp_rank.setValue(5)
        self.sp_rank.setPrefix("Top "); self.sp_rank.setSuffix(" moments")
        self.ed_rank_title = QLineEdit()
        self.ed_rank_title.setPlaceholderText("Title (default: TOP N MOMENTS)")
        self.btn_rank = QPushButton("3. Make ranking video")
        self.btn_rank.clicked.connect(self.on_rank)
        rl.addWidget(self.sp_rank); rl.addWidget(self.ed_rank_title, 1); rl.addWidget(self.btn_rank)
        lv.addWidget(self.rank_box)

        btns = QHBoxLayout()
        self.btn_analyze = QPushButton("1. Analyze"); self.btn_analyze.clicked.connect(self.on_analyze)
        self.btn_export = QPushButton("2. Export clips"); self.btn_export.clicked.connect(self.on_export)
        self.btn_cancel = QPushButton("Cancel"); self.btn_cancel.clicked.connect(self.on_cancel)
        self.btn_folder = QPushButton("Open output folder"); self.btn_folder.clicked.connect(self.on_open_folder)
        self.btn_new = QPushButton("New video"); self.btn_new.clicked.connect(self.on_new)
        for b in (self.btn_analyze, self.btn_export, self.btn_cancel, self.btn_folder, self.btn_new):
            btns.addWidget(b)
        lv.addLayout(btns)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(6, QHeaderView.Stretch)
        for c in range(6):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        lv.addWidget(self.table, 1)

        self.progress = QProgressBar(); self.progress.setRange(0, 1); self.progress.setValue(0)
        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(2000)
        self.log.setMaximumHeight(110)
        lv.addWidget(self.progress); lv.addWidget(self.status); lv.addWidget(self.log)
        split.addWidget(left)

        # ---- right column: preview + edit
        right = QWidget()
        rv = QVBoxLayout(right)
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.video = QVideoWidget()
        self.video.setMinimumHeight(260)
        self.player.setVideoOutput(self.video)
        self.player.positionChanged.connect(self.on_position)
        self.player.errorOccurred.connect(
            lambda err, msg: self._set_status(f"Preview problem: {msg or err}. Export still works.", warn=True))
        rv.addWidget(self.video, 1)

        pr = QHBoxLayout()
        self.btn_play = QPushButton("Play / Pause"); self.btn_play.clicked.connect(self.on_play_pause)
        self.btn_replay = QPushButton("Replay clip"); self.btn_replay.clicked.connect(self.on_replay)
        self.pos_label = QLabel("0:00.00")
        pr.addWidget(self.btn_play); pr.addWidget(self.btn_replay); pr.addStretch(1); pr.addWidget(self.pos_label)
        rv.addLayout(pr)

        self.edit_box = QGroupBox("Selected clip")
        eg = QGridLayout(self.edit_box)
        self.sp_start = QDoubleSpinBox(); self.sp_end = QDoubleSpinBox()
        for sp in (self.sp_start, self.sp_end):
            sp.setDecimals(2); sp.setSingleStep(0.1); sp.setRange(0, 100000); sp.setSuffix(" s")
        self.btn_set_start = QPushButton("Playhead -> start"); self.btn_set_start.clicked.connect(lambda: self.sp_start.setValue(self.player.position() / 1000.0))
        self.btn_set_end = QPushButton("Playhead -> end"); self.btn_set_end.clicked.connect(lambda: self.sp_end.setValue(self.player.position() / 1000.0))
        eg.addWidget(QLabel("Start"), 0, 0); eg.addWidget(self.sp_start, 0, 1); eg.addWidget(self.btn_set_start, 0, 2)
        eg.addWidget(QLabel("End"), 1, 0); eg.addWidget(self.sp_end, 1, 1); eg.addWidget(self.btn_set_end, 1, 2)
        self.btn_trim = QPushButton("Apply trim"); self.btn_trim.clicked.connect(self.on_trim)
        self.btn_merge = QPushButton("Merge with next"); self.btn_merge.clicked.connect(self.on_merge)
        self.btn_delete = QPushButton("Delete"); self.btn_delete.clicked.connect(self.on_delete)
        self.btn_reexp = QPushButton("Re-export this clip"); self.btn_reexp.clicked.connect(self.on_reexport)
        eg.addWidget(self.btn_trim, 2, 0, 1, 2); eg.addWidget(self.btn_merge, 2, 2)
        eg.addWidget(self.btn_delete, 3, 0, 1, 2); eg.addWidget(self.btn_reexp, 3, 2)
        self.text_label = QLabel("")
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet("color: #666;")
        eg.addWidget(self.text_label, 4, 0, 1, 3)
        rv.addWidget(self.edit_box)
        split.addWidget(right)
        split.setSizes([760, 540])

        # Plan settings stay editable: changing one turns Export off until you press Analyze again.
        self.sp_min.valueChanged.connect(self._on_settings_changed)
        self.sp_max.valueChanged.connect(self._on_settings_changed)
        self.chk_tr.toggled.connect(self._on_settings_changed)
        self.cb_model.currentIndexChanged.connect(self._on_settings_changed)
        self.ed_lang.textChanged.connect(self._on_settings_changed)
        # Render options only change what Export does.
        self.cb_caps.currentIndexChanged.connect(self._on_render_opts_changed)
        self.cb_vert.currentIndexChanged.connect(self._on_render_opts_changed)
        self.sp_top.valueChanged.connect(self._on_top_changed)

    # ----------------------------------------------------------------- helpers
    def _busy(self):
        return self.worker is not None and self.worker.isRunning()

    def _update_buttons(self):
        busy = self._busy()
        has_src = self.src is not None
        has_plan = self.manifest is not None
        can_export = False
        if has_plan and not busy:
            args = self._make_args(plan_only=False)
            # judged with the CURRENT render options, so a clip exported with other options counts as pending
            pending = clipper.pending_clips(self.manifest, self.out_dir,
                                            clipper.render_signature(args), clipper.wants_sidecar(args))
            can_export = bool(pending) and self._settings_match(args)
        exported = bool(has_plan and self.out_dir is not None and any(
            c.get("done") and (self.out_dir / c["file"]).exists() for c in clipper.live_clips(self.manifest)))
        self.rank_box.setEnabled(exported and not busy)
        sel = self._selected() is not None
        self.btn_open.setEnabled(not busy)
        self.ed_url.setEnabled(not busy)       # always pasteable; Download explains if yt-dlp is missing
        self.btn_dl.setEnabled(not busy)
        self.btn_analyze.setEnabled(has_src and not busy)
        self.btn_export.setEnabled(can_export)
        self.btn_cancel.setEnabled(busy)
        self.btn_folder.setEnabled(self.out_dir is not None)
        self.btn_new.setEnabled(has_src and not busy)
        self.settings.setEnabled(not busy)
        self.btn_out.setEnabled(not has_plan)      # the output folder belongs to the plan
        self.export_opts.setEnabled(not busy)
        for w in (self.btn_trim, self.btn_merge, self.btn_delete, self.btn_reexp,
                  self.sp_start, self.sp_end, self.btn_set_start, self.btn_set_end):
            w.setEnabled(sel and not busy)
        c = self._selected()
        self.btn_delete.setText("Restore" if c and c.get("deleted") else "Delete")

    def _set_status(self, text, warn=False):
        self.status.setText(text)
        self.status.setStyleSheet("color: #c0392b;" if warn else "")

    def _on_mode_toggled(self, fast):
        self.fast_warn.setVisible(fast)
        self._sync_fast_controls(fast)
        self._on_render_opts_changed()

    def _sync_fast_controls(self, fast):
        """Fast mode copies the video untouched: burned captions and 9:16 are impossible there."""
        for i in range(self.cb_caps.count()):
            if self.cb_caps.itemData(i) in ("burn", "both"):
                self.cb_caps.model().item(i).setEnabled(not fast)
        if fast:
            if self.cb_caps.currentData() in ("burn", "both"):
                self.cb_caps.setCurrentIndex(self.cb_caps.findData("sidecar"))
            self.cb_vert.setCurrentIndex(self.cb_vert.findData("off"))
        self.cb_vert.setEnabled(not fast)

    def _settings_match(self, args=None):
        """True when the plan on screen was made with the settings currently in the window."""
        if self.manifest is None or self.src is None:
            return False
        args = args or self._make_args(plan_only=True)
        return clipper.plan_fingerprint(self.src, args) == self.manifest.get("fingerprint")

    def _on_settings_changed(self, *_):
        self._update_buttons()
        if self.manifest is not None and not self._busy():
            if self._settings_match():
                self._set_status("Settings match the current plan. Review, then press Export.")
            else:
                self._set_status("Settings changed: press Analyze to plan again (this discards your edits). "
                                 "Export is off until then.", warn=True)

    def _on_render_opts_changed(self, *_):
        if self.manifest is not None and not self._busy():
            self._populate(keep_selection=True)      # refreshes the Status column for the new options
        self._update_buttons()

    def _apply_highlights(self):
        """Make the clip list match the 'best clips' box (safe to call any time a plan exists)."""
        if self.manifest is None:
            return
        n = self.sp_top.value()
        try:
            if n > 0:
                pool = [c for c in self.manifest["clips"] if not (c.get("deleted") and not c.get("auto_dropped"))]
                if any(c.get("h_score") is None for c in pool):      # trimmed / merged clips lose their score
                    clipper.score_clips(self.manifest, self.out_dir)
                clipper.select_highlights(self.manifest, self.out_dir, top=n)
            else:
                clipper.show_all_clips(self.manifest, self.out_dir)
        except clipper.ClipperError as e:
            self._set_status(f"{e} They will be scored when you press Export.", warn=True)

    def _on_top_changed(self, *_):
        if self.manifest is not None and not self._busy():
            self._apply_highlights()
            self._populate(keep_selection=True)
        self._update_buttons()

    def _selected(self):
        if self.manifest is None:
            return None
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        idx = item.data(Qt.UserRole)
        return next((c for c in self.manifest["clips"] if c["index"] == idx), None)

    # ------------------------------------------------------------ file handling
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() or e.mimeData().hasText():
            e.acceptProposedAction()

    def dropEvent(self, e):
        if self._busy():
            return
        md = e.mimeData()
        urls = md.urls()
        if urls and urls[0].scheme() in ("http", "https"):          # a link dragged from a browser
            self.ed_url.setText(urls[0].toString())
            self.on_download()
        elif urls:
            self.set_source(Path(urls[0].toLocalFile()))
        elif md.hasText() and clipper.is_url(md.text().strip()):
            self.ed_url.setText(md.text().strip())
            self.on_download()

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.flv *.ts *.wmv);;All files (*)")
        if path:
            self.set_source(Path(path))

    def set_source(self, path):
        if path.suffix.lower() not in VIDEO_EXT or not path.is_file():
            QMessageBox.warning(self, "Not a video", f"'{path.name}' does not look like a video file.")
            return
        self._discard_plan()
        self.src = path
        self.out_dir = path.with_name(path.stem + "_clips")
        self.file_label.setText(str(path))
        self.out_label.setText(f"Output folder: {self.out_dir}")
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self._set_status("Press Analyze to find the cut points.")
        self._update_buttons()

    def on_download(self):
        if self._busy():
            return
        url = self.ed_url.text().strip()
        if not url:
            return
        if not clipper.is_url(url):
            QMessageBox.warning(self, "Not a link", "Paste a full web link, starting with http:// or https://")
            return
        if not clipper.ytdlp_available():
            QMessageBox.warning(self, "yt-dlp missing", "Downloading links needs yt-dlp in the Python that runs this app:\n\n    "
                                + sys.executable + " -m pip install yt-dlp\n\nThen restart the app.")
            return
        self.worker = DownloadWorker(url)
        self.worker.log.connect(self._on_log)
        self.worker.progress.connect(self._on_dl_progress)
        self.worker.done.connect(self._on_dl_done)
        self.worker.failed.connect(self._on_dl_failed)
        self.progress.setRange(0, 0)          # busy indicator until the first percentage arrives
        self._set_status("Downloading... (saved to " + str(clipper.default_download_dir()) + ")")
        self.worker.start()
        self._update_buttons()

    def _on_dl_progress(self, pct, text):
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
        self.progress.setValue(pct)
        if text:
            self._set_status(text)

    def _on_dl_done(self, path):
        self.worker.wait()                    # the thread is just returning; makes the buttons update correctly
        self.progress.setRange(0, 1); self.progress.setValue(0)
        self.ed_url.clear()
        path = Path(path)
        self.set_source(path)
        if self.src == path:
            self._set_status(f"Downloaded {path.name}. Press Analyze to find the cut points.")
        self._update_buttons()

    def _on_dl_failed(self, msg):
        self.worker.wait()
        self.progress.setRange(0, 1); self.progress.setValue(0)
        self._set_status(msg, warn=True)
        if not msg.startswith("Cancelled") and self.isVisible():
            QMessageBox.critical(self, "Download failed", msg)
        self._update_buttons()

    def on_rank(self):
        if self._busy() or self.manifest is None:
            return
        self.worker = RankWorker(self.out_dir, self.sp_rank.value(), self.ed_rank_title.text().strip())
        self.worker.log.connect(self._on_log)
        self.worker.progress.connect(self._on_rank_progress)
        self.worker.done.connect(self._on_rank_done)
        self.worker.failed.connect(self._on_rank_failed)
        self.progress.setRange(0, 0)
        self._set_status("Making the ranking video...")
        self.worker.start()
        self._update_buttons()

    def _on_rank_progress(self, done, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self._set_status(f"Ranking video: {done} of {total} clips drawn")

    def _on_rank_done(self, path):
        self.worker.wait()
        self.manifest = clipper.load_job(self.out_dir) or self.manifest   # clips may have been scored
        self.progress.setRange(0, 1); self.progress.setValue(1)
        self._set_status(f"Ranking video saved: {path}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self._update_buttons()

    def _on_rank_failed(self, msg):
        self.worker.wait()
        self.progress.setRange(0, 1); self.progress.setValue(0)
        self._set_status(msg, warn=True)
        if not msg.startswith("Cancelled") and self.isVisible():
            QMessageBox.critical(self, "Ranking video failed", msg)
        self._update_buttons()

    def on_choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder")
        if d:
            self.out_dir = Path(d)
            self.out_label.setText(f"Output folder: {self.out_dir}")

    def on_open_folder(self):
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.out_dir)))

    def _discard_plan(self):
        if self.out_dir and self.manifest is not None:
            clipper.cleanup_work(self.out_dir)
        self.manifest = None
        self.table.setRowCount(0)
        self.progress.setRange(0, 1); self.progress.setValue(0)
        self.text_label.setText("")

    def on_new(self):
        self._discard_plan()
        self.src = self.out_dir = None
        self.player.stop()
        self.file_label.setText("Drop a video here, paste a link below, or press Open")
        self.out_label.setText("Output folder: (next to the video)")
        self._set_status("Ready.")
        self._update_buttons()

    # ----------------------------------------------------------------- pipeline
    def _make_args(self, plan_only):
        a = clipper.build_parser().parse_args([str(self.src), "-o", str(self.out_dir)])
        a.min_len, a.clip_len = self.sp_min.value(), self.sp_max.value()
        a.skip_transcription = not self.chk_tr.isChecked()
        a.whisper_model = self.cb_model.currentText()
        a.language = self.ed_lang.text().strip() or None
        a.plan_only = plan_only
        a.keep_work = True          # keep the normalized copy between Analyze and Export
        a.fast = self.rb_fast.isChecked()
        a.captions = self.cb_caps.currentData() or "off"
        a.vertical = self.cb_vert.currentData() or "off"
        a.highlights = self.sp_top.value() or None
        return a

    def _start(self, args, exporting):
        self.exporting = exporting
        self.worker = Worker(args)
        self.worker.log.connect(self._on_log)
        self.worker.event.connect(self._on_event)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished_ok.connect(self._on_finished)
        self.progress.setRange(0, 0)       # busy indicator until we know the totals
        self._set_status("Exporting..." if exporting else "Analyzing... (Whisper can take a while on CPU)")
        self.worker.start()
        self._update_buttons()

    def on_analyze(self):
        if self.sp_min.value() > self.sp_max.value():
            QMessageBox.warning(self, "Check settings", "Shortest clip must not be longer than the longest clip.")
            return
        args = self._make_args(plan_only=True)
        if self.manifest is not None and not self._settings_match(args):
            r = QMessageBox.question(
                self, "Plan again?",
                "The settings changed, so the clips have to be planned again from scratch.\n\n"
                "This discards your manual edits and deletes the clips already exported in the output folder.\n\n"
                "Continue?")
            if r != QMessageBox.Yes:
                return
            clipper.purge_clips(self.out_dir)
            self._discard_plan()
        self._start(args, exporting=False)

    def on_export(self):
        if self.manifest is None:
            return
        args = self._make_args(plan_only=False)
        if not self._settings_match(args):
            QMessageBox.information(self, "Settings changed",
                                    "The settings no longer match the plan. Press Analyze to plan again first.")
            return
        if args.fast and (args.captions in ("burn", "both") or args.vertical != "off"):
            QMessageBox.warning(self, "Check export options",
                                "Fast mode cannot burn captions or reframe to 9:16. "
                                "Use Precise mode, or choose sidecar .srt captions and no reframing.")
            return
        self._apply_highlights()
        self._populate(keep_selection=True)
        self._start(args, exporting=True)

    def on_cancel(self):
        if self._busy():
            self.worker.cancel_event.set()
            self._set_status("Cancelling... (stops within a few seconds)")

    def _on_log(self, line):
        self.log.appendPlainText(line)

    def _on_event(self, e):
        stage = e.get("stage")
        if stage == "plan_ready":
            if not self.exporting:   # during export the UI already has the plan (and the worker keeps editing its copy)
                self.manifest = copy.deepcopy(e["manifest"])
                self._populate()
        elif stage == "export_start":
            self.progress.setRange(0, max(1, e["total"]))
            self.progress.setValue(e["done"])
        elif stage == "clip_done":
            self.progress.setValue(e["done"])
            self._set_status(f"Exported {e['done']} of {e['total']}")
            self._mark_row_done(e["index"])

    def _on_finished(self):
        self.manifest = clipper.load_job(self.out_dir) or self.manifest
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if self.exporting else 0)
        self._populate(keep_selection=True)
        if self.exporting:
            n = len(clipper.live_clips(self.manifest))
            self._set_status(f"Done: {n} clips in {self.out_dir}")
        else:
            hard = sum(1 for c in clipper.live_clips(self.manifest) if c["signal"] == "hard")
            msg = f"{len(clipper.live_clips(self.manifest))} clips planned."
            if hard:
                msg += f" {hard} end on a hard cut (orange): check those first."
            self._set_status(msg + " Review, then press Export. (Settings are locked now; press New video to change them.)")
        self._update_buttons()

    def _on_failed(self, msg):
        self.progress.setRange(0, 1); self.progress.setValue(0)
        if self.out_dir and self.src is not None:
            m = clipper.load_job(self.out_dir)
            # a plan left on disk by OTHER settings must not replace what the window shows
            if m and (self.exporting or m.get("fingerprint") == clipper.plan_fingerprint(
                    self.src, self._make_args(plan_only=True))):
                self.manifest = m
                self._populate(keep_selection=True)
        self._set_status(msg, warn=True)
        if not msg.startswith("Cancelled") and self.isVisible():
            QMessageBox.critical(self, "Something went wrong", msg)
        self._update_buttons()

    # --------------------------------------------------------------------- table
    def _status_of(self, c, sig, sidecar):
        if c.get("deleted"):
            return "Not in best" if c.get("auto_dropped") else "Deleted"
        if clipper.clip_state(c, self.out_dir, sig, sidecar) == "ok":
            return "Exported"
        if c.get("done") and (self.out_dir / c["file"]).exists():
            return "Out of date"        # exported earlier with other options: Export will redo it
        return "Pending"

    def _populate(self, keep_selection=False):
        keep = self._selected()["index"] if keep_selection and self._selected() else None
        args = self._make_args(plan_only=False)
        sig, sidecar = clipper.render_signature(args), clipper.wants_sidecar(args)
        clips = sorted(self.manifest["clips"], key=lambda c: c["start"])
        self._quiet += 1                      # rebuilding the table must not start the preview
        try:
            self.table.blockSignals(True)
            try:
                self.table.setRowCount(len(clips))
                sel_row = -1
                for r, c in enumerate(clips):
                    length = c["end"] - c["start"]
                    cells = [str(c["index"]), fmt(c["start"]), fmt(c["end"]), f"{length:.1f}s",
                             SIGNAL_LABEL.get(c["signal"], c["signal"]), self._status_of(c, sig, sidecar),
                             c.get("text", "")]
                    for col, text in enumerate(cells):
                        it = QTableWidgetItem(text)
                        if col == 0:
                            it.setData(Qt.UserRole, c["index"])
                        if c.get("deleted"):
                            it.setForeground(QColor("#999999"))
                            f = it.font(); f.setStrikeOut(True); it.setFont(f)
                        elif c["signal"] == "hard":
                            it.setBackground(QColor("#ffe0b2")); it.setForeground(QColor("#000000"))
                        self.table.setItem(r, col, it)
                    if keep == c["index"]:
                        sel_row = r
            finally:
                self.table.blockSignals(False)
            if sel_row >= 0:
                self.table.selectRow(sel_row)
            self.on_select(autoplay=False)
        finally:
            self._quiet -= 1

    def _mark_row_done(self, index):
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.data(Qt.UserRole) == index:
                self.table.item(r, 5).setText("Exported")

    # ------------------------------------------------------------ preview / edit
    def _on_table_selection(self):
        # a click / arrow key plays the clip; selection changes made by the code itself (after an edit,
        # a re-populate, ...) only refresh the fields
        self.on_select(autoplay=(self._quiet == 0))

    def on_select(self, autoplay=True):
        c = self._selected()
        if c is not None:
            dur = self.manifest.get("duration") or 100000
            self.sp_start.setMaximum(dur); self.sp_end.setMaximum(dur)
            self.sp_start.setValue(c["start"]); self.sp_end.setValue(c["end"])
            self.text_label.setText(c.get("text") or "(no transcript for this clip)")
            if autoplay:
                self._play_range(c["start"], c["end"])
        self._update_buttons()

    def _play_range(self, start, end):
        self.preview_end = end
        self.player.setPosition(int(start * 1000))
        self.player.play()

    def on_position(self, ms):
        self.pos_label.setText(fmt(ms / 1000.0))
        if self.preview_end is not None and ms >= self.preview_end * 1000:
            self.player.pause()
            self.preview_end = None

    def on_play_pause(self):
        self.preview_end = None
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def on_replay(self):
        c = self._selected()
        if c:
            self._play_range(c["start"], c["end"])

    def _after_edit(self, warnings=None):
        sel = self._selected()
        keep = sel["index"] if sel else None
        self.manifest = clipper.load_job(self.out_dir)
        self._populate_and_select(keep)
        if warnings:
            self._set_status(" ".join(warnings), warn=True)
        else:
            self._set_status("Saved. Press Export to apply your changes.")
        self._update_buttons()

    def _populate_and_select(self, index):
        clips = sorted(self.manifest["clips"], key=lambda c: c["start"])
        self._quiet += 1                      # re-selecting after an edit must not restart the preview
        try:
            self._populate()
            for r, c in enumerate(clips):
                if c["index"] == index:
                    self.table.selectRow(r)
                    break
        finally:
            self._quiet -= 1

    def _guard(self, fn):
        try:
            return fn()
        except clipper.ClipperError as e:
            QMessageBox.warning(self, "Can't do that", str(e))
            return None

    def on_trim(self):
        c = self._selected()
        if not c:
            return
        w = self._guard(lambda: clipper.edit_trim(self.manifest, self.out_dir, c["index"],
                                                  self.sp_start.value(), self.sp_end.value(),
                                                  self.sp_min.value(), self.sp_max.value()))
        if w is not None:
            self._after_edit(w)

    def on_merge(self):
        c = self._selected()
        if not c:
            return
        w = self._guard(lambda: clipper.edit_merge_next(self.manifest, self.out_dir, c["index"], self.sp_max.value()))
        if w is not None:
            self._after_edit(w)

    def on_delete(self):
        c = self._selected()
        if not c:
            return
        if c.get("deleted"):
            clipper.edit_restore(self.manifest, self.out_dir, c["index"])
        else:
            clipper.edit_delete(self.manifest, self.out_dir, c["index"])
        self._after_edit()

    def on_reexport(self):
        c = self._selected()
        if not c or c.get("deleted"):
            return
        clipper.edit_reexport(self.manifest, self.out_dir, c["index"])
        self._after_edit()
        self.on_export()

    # -------------------------------------------------------------------- close
    def closeEvent(self, e):
        if self._busy():
            r = QMessageBox.question(self, "Still working", "A job is running. Cancel it and close?")
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.worker.cancel_event.set()
            if not self.worker.wait(15000):
                # Still stopping (e.g. a Whisper model is loading). Closing now would destroy a running
                # thread and can crash, so hide the window and quit the moment the worker has really ended.
                QApplication.setQuitOnLastWindowClosed(False)
                self.worker.finished.connect(QApplication.instance().quit)
                self.hide()
                e.ignore()
                return
        if self.out_dir and self.manifest is not None:
            clipper.cleanup_work(self.out_dir)   # progress stays in manifest.json
        e.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()