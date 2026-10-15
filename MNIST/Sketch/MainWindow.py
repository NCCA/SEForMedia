# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.12.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import QCoreApplication, QMetaObject, QRect, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLCDNumber,
    QLabel,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QStatusBar,
    QWidget,
)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1085, 466)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.main_layout = QGridLayout(self.centralwidget)
        self.main_layout.setObjectName("main_layout")
        self.horizontalSpacer_2 = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.main_layout.addItem(self.horizontalSpacer_2, 0, 2, 1, 1)

        self.controls = QGroupBox(self.centralwidget)
        self.controls.setObjectName("controls")
        self.gridLayout = QGridLayout(self.controls)
        self.gridLayout.setObjectName("gridLayout")
        self.clear_button = QPushButton(self.controls)
        self.clear_button.setObjectName("clear_button")

        self.gridLayout.addWidget(self.clear_button, 2, 0, 1, 1)

        self.label = QLabel(self.controls)
        self.label.setObjectName("label")

        self.gridLayout.addWidget(self.label, 3, 0, 1, 1)

        self.image_label = QLabel(self.controls)
        self.image_label.setObjectName("image_label")

        self.gridLayout.addWidget(self.image_label, 0, 0, 1, 1)

        self.numbers = QLCDNumber(self.controls)
        self.numbers.setObjectName("numbers")

        self.gridLayout.addWidget(self.numbers, 0, 1, 1, 1)

        self.label_2 = QLabel(self.controls)
        self.label_2.setObjectName("label_2")

        self.gridLayout.addWidget(self.label_2, 5, 0, 1, 1)

        self.pen_size = QSlider(self.controls)
        self.pen_size.setObjectName("pen_size")
        self.pen_size.setMinimum(1)
        self.pen_size.setMaximum(30)
        self.pen_size.setValue(10)
        self.pen_size.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.pen_size, 3, 1, 1, 1)

        self.proc_type = QComboBox(self.controls)
        self.proc_type.addItem("")
        self.proc_type.addItem("")
        self.proc_type.setObjectName("proc_type")

        self.gridLayout.addWidget(self.proc_type, 5, 1, 1, 1)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.gridLayout.addLayout(self.gridLayout_2, 0, 2, 1, 1)

        self.label_3 = QLabel(self.controls)
        self.label_3.setObjectName("label_3")

        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)

        self.border_size = QSlider(self.controls)
        self.border_size.setObjectName("border_size")
        self.border_size.setMaximum(15)
        self.border_size.setValue(4)
        self.border_size.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.border_size, 4, 1, 1, 1)

        self.main_layout.addWidget(self.controls, 0, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.main_layout.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 1085, 24))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.proc_type.setCurrentIndex(1)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("MainWindow", "Sketch Digits", None)
        )
        self.controls.setTitle(QCoreApplication.translate("MainWindow", "Image", None))
        self.clear_button.setText(
            QCoreApplication.translate("MainWindow", "clear", None)
        )
        self.label.setText(QCoreApplication.translate("MainWindow", "Pen Size", None))
        self.image_label.setText(
            QCoreApplication.translate("MainWindow", "TextLabel", None)
        )
        self.label_2.setText(
            QCoreApplication.translate("MainWindow", "Image Processing", None)
        )
        self.proc_type.setItemText(
            0, QCoreApplication.translate("MainWindow", "Bounding Box", None)
        )
        self.proc_type.setItemText(
            1, QCoreApplication.translate("MainWindow", "Center Mass", None)
        )

        self.label_3.setText(
            QCoreApplication.translate("MainWindow", "Border Size", None)
        )

    # retranslateUi
