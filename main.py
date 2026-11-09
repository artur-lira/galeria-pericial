"""Galeria Pericial — ponto de entrada."""
import multiprocessing
import os
import sys


def main():
    multiprocessing.freeze_support()
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QPalette, QColor
    from galeria.ui import MainWindow
    from galeria import APP_NAME

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in [(QPalette.Window, "#262626"), (QPalette.WindowText, "#e6e6e6"),
                      (QPalette.Base, "#1e1e1e"), (QPalette.AlternateBase, "#2a2a2a"),
                      (QPalette.Text, "#e6e6e6"), (QPalette.Button, "#333333"),
                      (QPalette.ButtonText, "#e6e6e6"), (QPalette.Highlight, "#2d5c8f"),
                      (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#333333"),
                      (QPalette.ToolTipText, "#ffffff")]:
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        w.pending_folder = sys.argv[1]  # abre assim que o motor de análise estiver pronto
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
