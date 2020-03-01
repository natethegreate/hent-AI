# Feb 2020 - Nathan Cueto
# Attempt to remove screentones from input images (png) using blurring and sharpening
#
# import sys
# sys.path.append('/usr/local/lib/python2.7/site-packages')
from os import listdir
import tkinter as tk
# from tkinter import ttk
# from matplotlib import pyplot as plt
from tkinter import filedialog
from detector import Detector

versionNumber = '1.0'

# root window, hidden. Only 1 active window at a time
root = tk.Tk()
root.withdraw()

current_window = None
counter = 0

# 1 - no png files found
# 2 - no input dir
# 3 - no output dir
# 4 - write error
def error(errcode):
    # popup success message
    popup = Tk()
    popup.title('Error')
    switcher = {
        1: "Error: No .png files found",
        2: "Error: No input directory",
        3: "Error: No output directory",
        4: "Error: File write error"
    }

    label = Label(popup, text=switcher.get(errcode, "what"))
    label.pack(side=TOP, fill=X, pady=20)

    okbutton = Button(popup, text='Ok', command=popup.destroy)
    okbutton.pack()
    popup.mainloop()
    # popup error code

# function scans directory and returns genorator
def getfileList(dir):
    return (i for i in listdir(dir) if i.endswith('.png'))

# globals that hold directory strings
dtext = ""
otext = ""

# both functions used to get and set directories
def dnewdir():
    dtext = filedialog.askdirectory(title='Choose directory for input .pngs')
    dvar.set(dtext)

def onewdir():
    otext = filedialog.askdirectory(title='Choose directory for output .pngs')
    ovar.set(otext)

def bar_detect():
    bar_win = new_window()
    bar_win.title('Bar Detection')


    bar_win.mainloop()

def mosaic_detect():
    mos_win = new_window()
    mos_win.title('Mosaic Detection')


    mos_win.mainloop()

# sourced from https://stackoverflow.com/a/35486067
def new_window():
    global counter
    counter += 1

    window = replace_window(root)
    return window

# sourced from https://stackoverflow.com/a/35486067
def  replace_window(root):
    """Destroy current window, create new window"""
    global current_window
    if current_window is not None:
        current_window.destroy()
    current_window = tk.Toplevel(root)

    # if the user kills the window via the window manager,
    # exit the application.
    current_window.wm_protocol("WM_DELETE_WINDOW", root.destroy)
    return current_window

if __name__ == "__main__":
    title_window = new_window()
    title_window.title("hentAI v." + versionNumber)

    intro_label = tk.Label(title_window, text='Welcome to hentAI! Please select a censor type: ')
    intro_label.pack(pady=10)
    bar_button = tk.Button(title_window, text="Bar", command=bar_detect)
    bar_button.pack(pady=10)
    mosaic_button = tk.Button(title_window, text="Mosaic", command=mosaic_detect)
    mosaic_button.pack(pady=10)

    title_window.geometry("300x160")
    title_window.mainloop()

    pass
