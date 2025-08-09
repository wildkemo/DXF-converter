import os
from typing import Optional

from PyQt5 import QtWidgets, QtCore

from core.preprocessing import preprocess_to_pbm
from core.potrace_wrapper import ensure_potrace_available, potrace_to_svg
from core.dxf_writer import svg_to_dxf


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image → DXF Converter")
        self.resize(600, 200)

        self.input_edit = QtWidgets.QLineEdit()
        self.browse_btn = QtWidgets.QPushButton("Browse…")
        self.output_edit = QtWidgets.QLineEdit("output")

        self.threshold_spin = QtWidgets.QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(128)

        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setRange(0.1, 20.0)
        self.step_spin.setSingleStep(0.1)
        self.step_spin.setValue(1.5)

        self.flip_checkbox = QtWidgets.QCheckBox("Flip Y axis (CAD)")
        self.flip_checkbox.setChecked(True)

        self.convert_btn = QtWidgets.QPushButton("Convert")

        form = QtWidgets.QFormLayout()
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(self.input_edit)
        hb.addWidget(self.browse_btn)
        form.addRow("Input file or folder", hb)
        form.addRow("Output folder", self.output_edit)
        form.addRow("Threshold", self.threshold_spin)
        form.addRow("Sample step", self.step_spin)
        form.addRow(self.flip_checkbox)
        form.addRow(self.convert_btn)

        self.setLayout(form)

        self.browse_btn.clicked.connect(self.on_browse)
        self.convert_btn.clicked.connect(self.on_convert)

    def on_browse(self):
        dlg = QtWidgets.QFileDialog(self, "Select image or folder")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if dlg.exec_():
            files = dlg.selectedFiles()
            if files:
                self.input_edit.setText(";".join(files))

    def on_convert(self):
        try:
            ensure_potrace_available()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Potrace not found", str(e))
            return

        inputs = [p for p in self.input_edit.text().split(";") if p]
        if not inputs:
            QtWidgets.QMessageBox.warning(self, "No input", "Please select input file(s)")
            return
        output_dir = self.output_edit.text() or "output"
        os.makedirs(output_dir, exist_ok=True)

        threshold = self.threshold_spin.value()
        step = self.step_spin.value()
        flip = self.flip_checkbox.isChecked()

        converted = 0
        for path in inputs:
            try:
                base = os.path.splitext(os.path.basename(path))[0]
                from tempfile import TemporaryDirectory

                with TemporaryDirectory(prefix="img2dxf_") as tmp:
                    pbm = preprocess_to_pbm(path, tmp, threshold=threshold)
                    svg = os.path.join(tmp, base + ".svg")
                    potrace_to_svg(pbm, svg)
                    dxf = os.path.join(output_dir, base + ".dxf")
                    svg_to_dxf(svg, dxf, sample_step=step, flip_y=flip)
                converted += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Conversion failed", f"{path}: {e}")

        QtWidgets.QMessageBox.information(self, "Done", f"Converted {converted} file(s)")


def launch():
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec_()




