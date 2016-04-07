# coding=utf-8
import sys
from PyQt4 import QtGui, QtCore, uic


class RegisterDialog(QtGui.QDialog):
    send_register_signal = QtCore.pyqtSignal(dict)
    def __init__(self):
        super(RegisterDialog, self).__init__()
        uic.loadUi("register.ui", self)

        self.create_signal_slot()
        #r = self.exec_()

    def create_signal_slot(self):
        self.connect(self.loginpushButton, QtCore.SIGNAL('clicked()'), self.loginButtonSlot)
        self.connect(self.cancelpushButton, QtCore.SIGNAL('clicked()'), self.cancelButtonSlot)

    def loginButtonSlot(self):
        name = self.namelineEdit.text()
        gender = self.genderlineEdit.text()
        age = self.agelineEdit.text()

        send_message_dict = {"name": name, "gender": gender, "age": age}
        self.send_register_signal.emit(send_message_dict)
        self.namelineEdit.clear()
        self.genderlineEdit.clear()
        self.agelineEdit.clear()
        self.close()

    def cancelButtonSlot(self):
        self.close()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    demo = RegisterDialog()

