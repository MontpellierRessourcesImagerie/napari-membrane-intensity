from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QApplication, QMainWindow,
                            QTableWidget, QTableWidgetItem, QFileDialog)
from PyQt5.QtGui import QColor, QFont
import numpy as np
import csv
import math

class ResultsTable(QMainWindow):
    def __init__(self, data, name='Data Table', parent=None):
        super(ResultsTable, self).__init__(parent)
        self.exp_name = "untitled.csv"
        self.setWindowTitle(name)
        self.font = QFont()
        self.font.setFamily("Arial Unicode MS, Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji")
        self.init_ui()
        self.set_data(data)

    def init_ui(self):
        # Central widget
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)

        # Layout
        self.layout = QVBoxLayout(self.centralWidget)

        # Table
        self.table = QTableWidget()
        self.layout.addWidget(self.table)  # Add table to layout

        # Export Button
        self.exportButton = QPushButton('💾 Save as CSV')
        self.exportButton.setFont(self.font)
        self.exportButton.clicked.connect(self.export_data)
        self.layout.addWidget(self.exportButton)

    def set_data(self, data):
        # Assume we have some data structure holding CSV-like data
        columnHeaders = ['Column 1', 'Column 2', 'Column 3']
        rowHeaders = ['Row 1', 'Row 2']
        rowData = [['Row1-Col1', 'Row1-Col2', 'Row1-Col3'],
                   ['Row2-Col1', 'Row2-Col2', 'Row2-Col3']]

        self.table.setColumnCount(len(columnHeaders))
        self.table.setRowCount(len(rowData))
        self.table.setHorizontalHeaderLabels(columnHeaders)
        self.table.setVerticalHeaderLabels(rowHeaders)

        for row, data in enumerate(rowData):
            for column, value in enumerate(data):
                item = QTableWidgetItem(value)
                # Set background color for the cell
                item.setBackground(QColor(255, 255, 200))  # Light yellow background
                self.table.setItem(row, column, item)

    def set_exp_name(self, name):
        self.exp_name = ".".join(name.replace(" ", "-").split('.')[:-1]) + ".csv"

    def export_data(self):
        options = QFileDialog.Options()
        try:
            fileName, _ = QFileDialog.getSaveFileName(
                self, 
                "QFileDialog.getSaveFileName()", 
                self.exp_name,
                "CSV Files (*.csv);;All Files (*)", 
                options=options
            )
        except:
            fileName = None

        if not fileName:
            print("No file selected")
            return
        
        self.export_table_to_csv(fileName)

    def export_table_to_csv(self, filename: str):
        # Open a file in write mode
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            
            # Writing headers (optional)
            headers = [self.table.horizontalHeaderItem(i).text() if self.table.horizontalHeaderItem(i) is not None else "" for i in range(self.table.columnCount())]
            writer.writerow(headers)
            
            # Writing data
            for row in range(self.table.rowCount()):
                row_data = []
                for column in range(self.table.columnCount()):
                    item = self.table.item(row, column)
                    # Check if the cell is not empty
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append('')
                writer.writerow(row_data)


# ====> The first result table contains the visibility and the centroid.


class FrameWiseResultsTable(ResultsTable):
    def __init__(self, data, name, parent=None):
        super(FrameWiseResultsTable, self).__init__(data, name, parent)
        self.init_ui()
        self.set_data(data)

    def set_data(self, data):
        if len(data) == 0:
            print("No data to display in the table.")
            return
        # Setting headers for each box.
        headers = [
            'Mean ring intensity', 
            'Integrated ring intensity', 
            'Ring area', 
            'Mean inner intensity', 
            'Integrated inner intensity', 
            'Inner area'
        ]
        nHeaders = len(headers)
        columnHeaders = []
        for name in data[0].keys():
            for header in headers:
                columnHeaders.append(f'{header} ({name})')
        
        # Settings rows headers.
        rowHeaders = [str(i+1) for i in range(len(data))]
        
        self.table.setColumnCount(len(columnHeaders))
        self.table.setRowCount(len(rowHeaders))
        self.table.setHorizontalHeaderLabels(columnHeaders)
        self.table.setVerticalHeaderLabels(rowHeaders)

        # Filling the table.
        for frame in range(len(data)): # For each frame
            for i, (lbl, (mean_r, integrated_r, area_r, mean_i, integrated_i, area_i)) in enumerate(data[frame].items()):
                color = QColor(99, 99, 99, 100)
                item_area_r = QTableWidgetItem(str(round(area_r, 2)) if area_r >= 0.0 else "")
                item_integrated_r = QTableWidgetItem(str(round(integrated_r, 2)) if integrated_r >= 0.0 else "")
                item_mean_r = QTableWidgetItem(str(round(mean_r, 2)) if mean_r >= 0.0 else "")
                item_area_i = QTableWidgetItem(str(round(area_i, 2)) if area_i >= 0.0 else "")
                item_integrated_i = QTableWidgetItem(str(round(integrated_i, 2)) if integrated_i >= 0.0 else "")
                item_mean_i = QTableWidgetItem(str(round(mean_i, 2)) if mean_i >= 0.0 else "")
                item_area_i.setBackground(color)
                item_integrated_r.setBackground(color)
                item_mean_i.setBackground(color)
                item_area_r.setBackground(color)
                item_integrated_i.setBackground(color)
                item_mean_r.setBackground(color)
                self.table.setItem(frame, i * 6 + 0, item_mean_r)
                self.table.setItem(frame, i * 6 + 1, item_integrated_r)
                self.table.setItem(frame, i * 6 + 2, item_area_r)
                self.table.setItem(frame, i * 6 + 3, item_mean_i)
                self.table.setItem(frame, i * 6 + 4, item_integrated_i)
                self.table.setItem(frame, i * 6 + 5, item_area_i)

        self.table.resizeColumnsToContents()
