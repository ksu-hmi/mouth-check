    #alert made by Megan to alert decay to clinician
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
#end of edit to code by Megan