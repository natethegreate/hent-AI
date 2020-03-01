# Feb 2020 - Nathan Cueto
# Attempt to remove screentones from input images (png) using blurring and sharpening
#
# import sys
# sys.path.append('/usr/local/lib/python2.7/site-packages')
from os import listdir
from tkinter import *
# from tkinter import ttk
# from matplotlib import pyplot as plt
from tkinter import filedialog
import shutil

versionNumber = '1.0'
weights_path = 'weights.h5' # should call it weights.h5 in main dir

# tkinter UI globals for window tracking. Sourced from https://stackoverflow.com/a/35486067
# root window, hidden. Only 1 active window at a time
root = Tk()
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

# Just build the video at the root location
# image_path is default the DCP decensor output
def hentAI_video_create(video_path=None, dcp_dir=None):
    # video create does not use self, so create dummy class
    from detector import Detector
    video_instance = Detector(weights_path='')
    video_instance.video_create(image_path=video_path, dcp_path=dcp_dir)
    print('Process complete!')
    popup = Tk()
    popup.title('Success!')
    label = Label(popup, text='Video Created!.')
    label.pack(side=TOP, fill=X, pady=20, padx=10)
    okbutton = Button(popup, text='Ok', command=popup.destroy)
    okbutton.pack()
    popup.mainloop()

def hentAI_detection(dcp_dir=None, in_path=None, is_mosaic=False, is_video=False):
    #Import the big guns here. It can take a while for tensorflow, and a laggy initial bringup can look sketchy tbh
    from detector import Detector

    # hent_win = new_window()
    # info_label = Label(hent_win, text="Beginning detection")
    # info_label.pack(padx=10,pady=10)
    # hent_win.mainloop()
    # repace these with catches and use error function
    assert dcp_dir
    assert in_path


    # print('Initializing Detector class')
    detect_instance = Detector(weights_path=weights_path)
    # print('loading weights')
    detect_instance.load_weights()
    if(is_mosaic == True):
        # Copy input folder to decensor_input_original. NAMES MUST MATCH for DCP
        print('copying inputs into input_original dcp folder')
        # print(in_path)
        # print(listdir(in_path))
        for file in listdir(in_path):
            # kinda dumb but check if same file
            shutil.copy(in_path + '/' + file, dcp_dir + '/decensor_input_original/')

    # Run detection
    if(is_video==True):
        print('running video detection')
        detect_instance.run_on_folder(input_folder=in_path, output_folder=dcp_dir+'/decensor_input/', is_video=True, orig_video_folder=dcp_dir + '/decensor_input_original/')
    else:
        print('running detection, outputting to dcp input')
        detect_instance.run_on_folder(input_folder=in_path, output_folder=dcp_dir+'/decensor_input/', is_video=False)



    print('Process complete!')
    popup = Tk()
    popup.title('Success!')
    label = Label(popup, text='Process executed successfully! Now you close the program and run DeepCreamPy.')
    label.pack(side=TOP, fill=X, pady=20, padx=10)
    num_jpgs = detect_instance.get_non_png()
    if(num_jpgs > 0):
        label2 = Label(popup, text= str(num_jpgs) + " files are NOT in .png format, and were not processed.\nPlease convert jpgs to pngs.")
        label2.pack(side=TOP, fill=X, pady=10, padx=5)
    okbutton = Button(popup, text='Ok', command=popup.destroy)
    okbutton.pack()
    popup.mainloop()

# function scans directory and returns generator
def getfileList(dir):
    return (i for i in listdir(dir) if i.endswith('.png'))

# globals that hold directory strings
dtext = ""
otext = ""


# both functions used to get and set directories
def dcp_newdir():
    dtext = filedialog.askdirectory(title='Choose directory for DCP installation')
    dvar.set(dtext)

def input_newdir():
    otext = filedialog.askdirectory(title='Choose directory for input .pngs')
    ovar.set(otext)

def bar_detect():
    bar_win = new_window()
    bar_win.title('Bar Detection')

    # input image directory label, entry, and button
    o_label = Label(bar_win, text = 'Your own input image (.png) folder: ')
    o_label.grid(row=1, padx=20 ,pady=10)
    o_entry = Entry(bar_win, textvariable=ovar)
    o_entry.grid(row=1, column=1)
    out_button = Button(bar_win, text="Browse", command=input_newdir)
    out_button.grid(row=1, column=2)

    # Entry for DCP installation
    d_label = Label(bar_win, text = 'DCP install folder (usually called dist1): ')
    d_label.grid(row=2, padx=20, pady=10)
    d_entry = Entry(bar_win, textvariable = dvar)
    d_entry.grid(row=2, column=1, padx=20)
    dir_button = Button(bar_win, text="Browse", command=dcp_newdir)
    dir_button.grid(row=2, column=2, padx=20)

    go_button = Button(bar_win, text="Go!", command = lambda: hentAI_detection(dcp_dir=d_entry.get(), in_path=o_entry.get(), is_mosaic=False, is_video=False))
    go_button.grid( columnspan=2, pady=10)

    bar_win.mainloop()

def mosaic_detect():
    mos_win = new_window()
    mos_win.title('Mosaic Detection')

    # input image directory label, entry, and button
    o_label = Label(mos_win, text = 'Your own input image (.png) folder: ')
    o_label.grid(row=1, padx=20 ,pady=10)
    o_entry = Entry(mos_win, textvariable=ovar)
    o_entry.grid(row=1, column=1)
    out_button = Button(mos_win, text="Browse", command=input_newdir)
    out_button.grid(row=1, column=2)

    # Entry for DCP installation
    d_label = Label(mos_win, text = 'DCP install folder (usually called dist1): ')
    d_label.grid(row=2, padx=20, pady=20)
    d_entry = Entry(mos_win, textvariable = dvar)
    d_entry.grid(row=2, column=1, padx=20)
    dir_button = Button(mos_win, text="Browse", command=dcp_newdir)
    dir_button.grid(row=2, column=2, padx=20)

    go_button = Button(mos_win, text="Go!", command = lambda: hentAI_detection(dcp_dir=d_entry.get(), in_path=o_entry.get(), is_mosaic=True, is_video=False))
    go_button.grid( columnspan=2, pady=10)

    mos_win.mainloop()

def video_detect():
    vid_win = new_window()
    vid_win.title('Video Detection (Experimental)')

    # input video(s) directory label, entry, and button
    o_label = Label(vid_win, text = 'Your own input video (???) folder: ')
    o_label.grid(row=1, padx=20 ,pady=10)
    o_entry = Entry(vid_win, textvariable=ovar)
    o_entry.grid(row=1, column=1)
    out_button = Button(vid_win, text="Browse", command=input_newdir)
    out_button.grid(row=1, column=2)

    # Entry for DCP installation
    d_label = Label(vid_win, text = 'DCP install folder (usually called dist1): ')
    d_label.grid(row=2, padx=20, pady=20)
    d_entry = Entry(vid_win, textvariable = dvar)
    d_entry.grid(row=2, column=1, padx=20)
    dir_button = Button(vid_win, text="Browse", command=dcp_newdir)
    dir_button.grid(row=2, column=2, padx=20)

    go_button = Button(vid_win, text="Begin Detection!", command = lambda: hentAI_detection(dcp_dir=d_entry.get(), in_path=o_entry.get(), is_mosaic=True, is_video=True))
    go_button.grid(row=3, columnspan=2, pady=5)

    vid_label = Label(vid_win, text= 'If you finished the video uncensoring, put images from DCP output back into video format. Check README for usage.')
    vid_label.grid(row=4, pady=5, padx=4)
    vid_button = Button(vid_win, text='Begin Video Maker!', command = lambda: hentAI_video_create(dcp_dir=d_entry.get(), video_path=o_entry.get()))
    vid_button.grid(row=5, pady=5, padx=10)

    vid_win.mainloop()

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
    current_window = Toplevel(root)

    # if the user kills the window via the window manager,
    # exit the application.
    current_window.wm_protocol("WM_DELETE_WINDOW", root.destroy)
    return current_window

if __name__ == "__main__":
    title_window = new_window()
    title_window.title("hentAI v." + versionNumber)

    dvar = StringVar(root)
    ovar = StringVar(root)

    intro_label = Label(title_window, text='Welcome to hentAI! Please select a censor type: ')
    intro_label.pack(pady=10)
    bar_button = Button(title_window, text="Bar", command=bar_detect)
    bar_button.pack(pady=10)
    mosaic_button = Button(title_window, text="Mosaic", command=mosaic_detect)
    mosaic_button.pack(pady=10)
    video_button = Button(title_window, text='Video (Experimental)', command=video_detect)
    video_button.pack(pady=10, padx=10)

    title_window.geometry("300x200")
    title_window.mainloop()

    pass
