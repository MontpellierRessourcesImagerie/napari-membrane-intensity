from qtpy.QtCore import QObject
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import numpy as np

class QtSegmentCells(QObject):
    
    finished = pyqtSignal()

    def __init__(self, scw):
        super().__init__()
        self.scw = scw

    def run(self):
        self.scw.run()
        self.finished.emit()


class QtTrackCells(QObject):

    finished = pyqtSignal()

    def __init__(self, tcw):
        super().__init__()
        self.tcw = tcw

    def run(self):
        self.tcw.run()
        self.finished.emit()


class QtMeasureMembranes(QObject):

    finished = pyqtSignal()

    def __init__(self, miw):
        super().__init__()
        self.miw = miw

    def run(self):
        self.miw.run()
        self.finished.emit()