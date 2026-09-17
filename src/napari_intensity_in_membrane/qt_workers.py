from qtpy.QtCore import QObject, Signal


class QtSegmentCells(QObject):
    
    finished = Signal()

    def __init__(self, scw):
        super().__init__()
        self.scw = scw

    def run(self):
        self.scw.run()
        self.finished.emit()


class QtTrackCells(QObject):

    finished = Signal()

    def __init__(self, tcw):
        super().__init__()
        self.tcw = tcw

    def run(self):
        self.tcw.run()
        self.finished.emit()


class QtRemoveOutlierIntensities(QObject):

    finished = Signal()

    def __init__(self, roi):
        super().__init__()
        self.roi = roi

    def run(self):
        self.roi.run()
        self.finished.emit()


class QtMeasureMembranes(QObject):

    finished = Signal()

    def __init__(self, miw):
        super().__init__()
        self.miw = miw

    def run(self):
        self.miw.run()
        self.finished.emit()