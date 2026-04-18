import json
import sys
from pathlib import Path

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


def extract_json_object(text: str):
    s = (text or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        maybe = s[start : end + 1]
        try:
            obj = json.loads(maybe)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def load_val_infer(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)
    return records


def summarize_actions(actions):
    if not isinstance(actions, list):
        return {"num_frames": 0}
    n = len(actions)
    if n == 0:
        return {"num_frames": 0}
    gripper = [a[6] for a in actions if isinstance(a, list) and len(a) >= 7]
    stats = {"count": len(gripper)}
    if gripper:
        stats.update(
            {
                "min": float(min(gripper)),
                "max": float(max(gripper)),
                "mean": float(sum(gripper) / len(gripper)),
                "transitions": int(sum(1 for i in range(1, len(gripper)) if gripper[i] != gripper[i - 1])),
            }
        )
    return {"num_frames": n, "gripper_stats": stats}


def load_instruction_and_actions(sample_id: str):
    p = Path("~/autodl-tmp/data/libero/processed_data").expanduser() / sample_id
    meta = p / "metadata.json"
    if meta.exists():
        data = json.loads(meta.read_text(encoding="utf-8"))
        instruction = str(data.get("instruction", ""))
        actions = data.get("actions_sequence", [])
        return instruction, summarize_actions(actions)
    return "", {"num_frames": 0}


class VideoPlayer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.play_btn = QtWidgets.QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self.toggle_play)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self.seek)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(self.play_btn)
        ctrl.addWidget(self.slider)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.label, stretch=1)
        lay.addLayout(ctrl)
        self.cap = None
        self.total_frames = 0
        self.fps = 25.0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.cur_frame = 0
        self.path = ""

    def set_video(self, path: str):
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                self.label.setText("无法打开视频")
                return
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.cur_frame = 0
            self.path = path
            interval = int(1000.0 / max(self.fps, 1.0))
            self.timer.start(interval)
            self.play_btn.setText("⏸")
        except Exception:
            self.label.setText("视频加载失败")

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("▶")
        else:
            interval = int(1000.0 / max(self.fps, 1.0))
            self.timer.start(interval)
            self.play_btn.setText("⏸")

    def next_frame(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.timer.stop()
            self.play_btn.setText("▶")
            return
        self.cur_frame += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.label.setPixmap(pix.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        # update slider
        self.slider.blockSignals(True)
        if self.total_frames > 0:
            self.slider.setValue(int(self.cur_frame * 1000 / self.total_frames))
        self.slider.blockSignals(False)

    def seek(self, v):
        if not self.cap or self.total_frames <= 0:
            return
        target = int(v * self.total_frames / 1000)
        target = max(0, min(target, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.cur_frame = target


class ScrollCard(QtWidgets.QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = QtWidgets.QLabel(title)
        self.title.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.content = QtWidgets.QTextEdit()
        self.content.setReadOnly(True)
        self.zoom = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.zoom.setRange(10, 24)
        self.zoom.setValue(13)
        self.zoom.valueChanged.connect(self.on_zoom)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(QtWidgets.QLabel("字体大小"))
        top.addWidget(self.zoom)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.content)

    def set_text(self, text: str):
        self.content.setPlainText(text)

    def on_zoom(self, v):
        f = self.content.font()
        f.setPointSize(v)
        self.content.setFont(f)


class EvidenceList(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = QtWidgets.QLabel("Model Evidence")
        self.title.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.listw = QtWidgets.QListWidget()
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addWidget(self.listw)

    def set_items(self, items):
        self.listw.clear()
        for i in items or []:
            self.listw.addItem(str(i))


class InfoGrid(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = QtWidgets.QLabel("Model Decision")
        self.title.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.decision = QtWidgets.QLabel("")
        self.issue = QtWidgets.QLabel("")
        self.score = QtWidgets.QLabel("")
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Decision"), 0, 0)
        grid.addWidget(self.decision, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Issue Focus"), 1, 0)
        grid.addWidget(self.issue, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Quality Score"), 2, 0)
        grid.addWidget(self.score, 2, 1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addLayout(grid)

    def set_info(self, decision: str, issue: str, score):
        self.decision.setText(str(decision or ""))
        self.issue.setText(str(issue or ""))
        self.score.setText("" if score is None else str(score))


class Surface(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LIBERO VLM Evaluation Viewer")
        self.setMinimumSize(1200, 800)
        self.index = 0
        self.records = load_val_infer(Path("~/autodl-fs/output/libero/SFT_eval/val_infer.jsonl").expanduser())
        self.video = VideoPlayer()
        self.instruction = ScrollCard("Instruction")
        self.actions = ScrollCard("Action Summary")
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.video, stretch=3)
        left.addWidget(self.instruction, stretch=1)
        left.addWidget(self.actions, stretch=1)
        self.evidence = EvidenceList()
        self.info = InfoGrid()
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.info, stretch=1)
        right.addWidget(self.evidence, stretch=2)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addLayout(left, stretch=3)
        layout.addLayout(right, stretch=2)
        self.prev_btn = QtWidgets.QPushButton("← 上一个")
        self.next_btn = QtWidgets.QPushButton("下一个 →")
        self.prev_btn.clicked.connect(self.prev)
        self.next_btn.clicked.connect(self.next)
        nav = QtWidgets.QHBoxLayout()
        nav.addStretch()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)
        self.apply_style()
        self.load_sample(0)

    def apply_style(self):
        self.setStyleSheet(
            """
            QWidget { background-color: #121212; color: #e0e0e0; }
            QGroupBox { border: 1px solid #333; margin-top: 10px; }
            QPushButton { background-color: #1f1f1f; padding: 6px 10px; border: 1px solid #444; border-radius: 4px; }
            QPushButton:hover { background-color: #2a2a2a; }
            QListWidget { background-color: #1a1a1a; }
            QTextEdit { background-color: #1a1a1a; }
            """
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Right:
            self.next()
        elif event.key() == QtCore.Qt.Key_Left:
            self.prev()
        else:
            super().keyPressEvent(event)

    def prev(self):
        if self.index > 0:
            self.index -= 1
            self.load_sample(self.index)

    def next(self):
        if self.index + 1 < len(self.records):
            self.index += 1
            self.load_sample(self.index)

    def parse_model_json(self, record):
        pred = extract_json_object(record.get("pred_text", "")) or {}
        if not pred:
            pred = extract_json_object(record.get("ground_truth", "")) or {}
        decision = pred.get("decision", "")
        issue = pred.get("issue_focus", "")
        score = pred.get("quality_score", None)
        evidence = pred.get("evidence", [])
        return decision, issue, score, evidence

    def load_sample(self, idx: int):
        rec = self.records[idx]
        vids = rec.get("videos", [])
        if vids:
            self.video.set_video(vids[0])
        sample_id = rec.get("id", "")
        instruction, action_summary = load_instruction_and_actions(sample_id)
        self.instruction.set_text(instruction)
        self.actions.set_text(json.dumps(action_summary, ensure_ascii=False, indent=2))
        decision, issue, score, evidence = self.parse_model_json(rec)
        self.info.set_info(decision, issue, score)
        self.evidence.set_items(evidence)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Surface()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
