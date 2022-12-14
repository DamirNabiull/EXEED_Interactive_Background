# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Menu.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MenuMainWindow(object):
    def setupUi(self, MenuMainWindow):
        MenuMainWindow.setObjectName("MenuMainWindow")
        MenuMainWindow.setEnabled(True)
        MenuMainWindow.resize(1080, 1920)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MenuMainWindow.sizePolicy().hasHeightForWidth())
        MenuMainWindow.setSizePolicy(sizePolicy)
        MenuMainWindow.setFocusPolicy(QtCore.Qt.NoFocus)
        MenuMainWindow.setAnimated(True)
        MenuMainWindow.setDocumentMode(False)
        self.centralwidget = QtWidgets.QWidget(MenuMainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.menuBackground = QtWidgets.QLabel(self.centralwidget)
        self.menuBackground.setGeometry(QtCore.QRect(0, 0, 1080, 1920))
        self.menuBackground.setText("")
        self.menuBackground.setPixmap(QtGui.QPixmap("design/assets/MenuBackground.png"))
        self.menuBackground.setObjectName("menuBackground")
        self.menuStartButton = QtWidgets.QPushButton(self.centralwidget)
        self.menuStartButton.setGeometry(QtCore.QRect(347, 698, 386, 108))
        self.menuStartButton.setStyleSheet("background-image: url(design/assets/MenuStartButton.png);\n"
"border-style: outset;\n"
"border-width: 0px;\n"
"border-color: rgba(0,0,0,0);")
        self.menuStartButton.setText("")
        self.menuStartButton.setObjectName("menuStartButton")
        MenuMainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MenuMainWindow)
        QtCore.QMetaObject.connectSlotsByName(MenuMainWindow)

    def retranslateUi(self, MenuMainWindow):
        _translate = QtCore.QCoreApplication.translate
        MenuMainWindow.setWindowTitle(_translate("MenuMainWindow", "MainWindow"))
