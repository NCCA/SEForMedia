# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.12.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import QCoreApplication, QMetaObject, QRect
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenuBar,
    QSizePolicy,
    QSpacerItem,
    QStatusBar,
    QWidget,
)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.resize(723, 402)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.grid_layout = QGridLayout(self.centralwidget)
        self.grid_layout.setObjectName("grid_layout")
        self.predicted_letter = QLabel(self.centralwidget)
        self.predicted_letter.setObjectName("predicted_letter")
        font = QFont()
        font.setPointSize(100)
        self.predicted_letter.setFont(font)
        self.predicted_letter.setFrameShape(QFrame.Box)

        self.grid_layout.addWidget(self.predicted_letter, 0, 1, 1, 1)

        self.asl_label = QLabel(self.centralwidget)
        self.asl_label.setObjectName("asl_label")

        self.grid_layout.addWidget(self.asl_label, 2, 1, 1, 1)

        self.verticalSpacer = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.grid_layout.addItem(self.verticalSpacer, 0, 0, 1, 1)

        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout = QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.input_image = QLabel(self.groupBox)
        self.input_image.setObjectName("input_image")

        self.horizontalLayout.addWidget(self.input_image)

        self.cropped_image = QLabel(self.groupBox)
        self.cropped_image.setObjectName("cropped_image")

        self.horizontalLayout.addWidget(self.cropped_image)

        self.grid_layout.addWidget(self.groupBox, 2, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 723, 24))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("MainWindow", "Capture", None)
        )
        self.predicted_letter.setText("")
        self.asl_label.setText(
            QCoreApplication.translate("MainWindow", "TextLabel", None)
        )
        self.groupBox.setTitle("")
        self.input_image.setText(
            QCoreApplication.translate("MainWindow", "TextLabel", None)
        )
        self.cropped_image.setText(
            QCoreApplication.translate("MainWindow", "TextLabel", None)
        )

    # retranslateUi
