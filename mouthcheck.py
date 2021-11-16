
# adding a pop up alert message box for okaying cavity check or warning cavities are detected
from tkinter import messagebox, Tk


def alert(title, message, kind='info', hidemain=True):
    if kind not in ('no caries detected'):
        raise ValueError('Diagnosis: Good')

    show_method = getattr(messagebox, 'show{}'.format(kind))
    show_method(title, message)


if __name__ == '__main__':
    Tk().withdraw()
    alert('No caries detected')
    alert('Caries detected', kind='warning!')
