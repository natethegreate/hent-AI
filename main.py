# Feb 2020 - Nathan Cueto
# Main function for UI and uses Detector class

import sys
# sys.path.append('C:\Users\natha\Anaconda3\envs\tensorflow\lib\site-packages')
from os import listdir, system
from tkinter import *
# from tkinter import Label, Entry, Button, Tk, StringVar, TOP, X, Toplevel
# from tkinter import ttk
# from matplotlib import pyplot as plt
import subprocess
from tkinter import filedialog
import shutil
from detector import Detector

versionNumber = '1.6.1'
weights_path = 'weights.h5' # should call it weights.h5 in main dir

# tkinter UI globals for window tracking. Sourced from https://stackoverflow.com/a/35486067
# root window, hidden. Only 1 active window at a time
root = Tk()
root.withdraw()
current_window = None
counter = 0

# for compiling fix
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# 1 - no png files found
# 2 - no input dir
# 3 - no DCP dir
# 4 - write error
# 5 - DCP directory invalid
def error(errcode):
    # popup success message
    popup = Tk()
    popup.title('Error')
    switcher = {
        1: "Error: No .png files found",
        2: "Error: No input directory",
        3: "Error: No DCP directory",
        4: "Error: File write error",
        5: "Error: DCP directory is invalid"
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
    if dcp_dir==None:
        error(5)
    if video_path==None:
        error(2)

    video_instance = Detector(weights_path='')

    loader = Tk()
    loader.title('Running video creator')
    load_label = Label(loader, text='Now creating video. Please wait a moment.')
    load_label.pack(side=TOP, fill=X, pady=10, padx=20)
    loader.update()

    video_instance.video_create(image_path=video_path, dcp_path=dcp_dir)
    loader.destroy()
    print('Process complete!')
    popup = Tk()
    popup.title('Success!')
    label = Label(popup, text='Video Created!.')
    label.pack(side=TOP, fill=X, pady=20, padx=10)
    okbutton = Button(popup, text='Ok', command=popup.destroy)
    okbutton.pack()
    popup.mainloop()

def hentAI_detection(dcp_dir=None, in_path=None, is_mosaic=False, is_video=False, force_jpg=False):
    # TODO: Create new window? Can show loading bar
    # hent_win = new_window()
    # info_label = Label(hent_win, text="Beginning detection")
    # info_label.pack(padx=10,pady=10)
    # hent_win.mainloop()

    if dcp_dir==None:
        error(5)
    if in_path==None:
        error(2)

    # print(force_jpg) #debug
    # return    

    # print('Initializing Detector class')
    if(is_mosaic == True and is_video==False):
        # Copy input folder to decensor_input_original. NAMES MUST MATCH for DCP
        print('copying inputs into input_original dcp folder')
        # print(in_path)
        # print(listdir(in_path))
        for file in listdir(in_path):
            # kinda dumb but check if same file
            if force_jpg==True:
                try:
                    shutil.copy(in_path + '/' + file, dcp_dir + '/decensor_input_original/' + file[:-4] + '.png')
                except:
                    print("ERROR in hentAI_detection: Mosaic copy + png conversion to decensor_input_original failed!")
                    return
            else:
                shutil.copy(in_path + '/' + file, dcp_dir + '/decensor_input_original/')

    # Run detection
    if(is_video==True):
        print('Running video detection')
        loader = Tk()
        loader.title('Running detections')
        load_label = Label(loader, text='Now running detections. This can take around a minute or so per image. Please wait')
        load_label.pack(side=TOP, fill=X, pady=10, padx=20)
        loader.update()
        detect_instance.run_on_folder(input_folder=in_path, output_folder=dcp_dir+'/decensor_input/', is_video=True, orig_video_folder=dcp_dir + '/decensor_input_original/') #no jpg for video detect
        loader.destroy()
    else:
        print('Running detection, outputting to dcp input')
        loader = Tk()
        loader.title('Running detections')
        load_label = Label(loader, text='Now running detections. This can take around a minute or so per image. Please wait')
        load_label.pack(side=TOP, fill=X, pady=10, padx=20)
        loader.update()
        detect_instance.run_on_folder(input_folder=in_path, output_folder=dcp_dir+'/decensor_input/', is_video=False, force_jpg=force_jpg, is_mosaic=is_mosaic)
        loader.destroy()


    # Announce completion, TODO: offer to run DCP from DCP directory
    print('Process complete!')
    popup = Tk()
    popup.title('Success!')
    label = Label(popup, text='Process executed successfully! Now you can run DeepCreamPy.')
    label.pack(side=TOP, fill=X, pady=20, padx=10)
    num_jpgs = detect_instance.get_non_png()
    # Popup for unprocessed jpgs
    if(num_jpgs > 0 and force_jpg==False):
        label2 = Label(popup, text= str(num_jpgs) + " files are NOT in .png format, and were not processed.\nPlease convert jpgs to pngs.")
        label2.pack(side=TOP, fill=X, pady=10, padx=5)
    # dcprun = Button(popup, text='Run DCP (Only if you have the .exe)', command= lambda: run_dcp(dcp_dir))
    # dcprun.pack(pady=10)
    okbutton = Button(popup, text='Ok', command=popup.destroy)
    okbutton.pack()
    popup.mainloop()

# helper function to call TGAN folder function. 
def hentAI_TGAN(in_path=None, is_video=False, force_jpg=False):
    print("Starting TGAN detection and decensor")
    loader = Tk()
    loader.title('Running TecoGAN')
    load_label = Label(loader, text='Now running decensor. This can take a while. Please wait')
    load_label.pack(side=TOP, fill=X, pady=10, padx=20)
    loader.update()
    detect_instance.run_TGAN(in_path = in_path, is_video = is_video, force_jpg = force_jpg)
    loader.destroy()

    print('Process complete!')
    popup = Tk()
    popup.title('Success!')
    label = Label(popup, text='Process executed successfully!')
    label.pack(side=TOP, fill=X, pady=20, padx=10)
    num_jpgs = detect_instance.get_non_png()
    # Popup for unprocessed jpgs
    # if(num_jpgs > 0 and force_jpg==False):
    #     label2 = Label(popup, text= str(num_jpgs) + " files are NOT in .png format, and were not processed.\nPlease convert jpgs to pngs.")
    #     label2.pack(side=TOP, fill=X, pady=10, padx=5)
    # dcprun = Button(popup, text='Run DCP (Only if you have the .exe)', command= lambda: run_dcp(dcp_dir))
    # dcprun.pack(pady=10)
    okbutton = Button(popup, text='Ok', command=popup.destroy)
    okbutton.pack()
    popup.mainloop()

# globals that hold directory strings
dtext = ""
otext = ""


# both functions used to get and set directories
def dcp_newdir():
    dtext = filedialog.askdirectory(title='Choose directory for DeepCreamPy installation')
    dvar.set(dtext)

def input_newdir():
    otext = filedialog.askdirectory(title='Choose directory for input .pngs')
    ovar.set(otext)

def bar_detect():
    bar_win = new_window()
    bar_win.title('Bar Detection')

    # input image directory label, entry, and button
    o_label = Label(bar_win, text = 'Your own input image folder: ')
    o_label.grid(row=1, padx=20 ,pady=10)
    o_entry = Entry(bar_win, textvariable=ovar)
    o_entry.grid(row=1, column=1)
    out_button = Button(bar_win, text="Browse", command=input_newdir)
    out_button.grid(row=1, column=2)

    # Entry for DCP installation
    d_label = Label(bar_win, text = 'DeepCreamPy install folder (usually called dist1): ')
    d_label.grid(row=2, padx=20, pady=10)
    d_entry = Entry(bar_win, textvariable = dvar)
    d_entry.grid(row=2, column=1, padx=20)
    dir_button = Button(bar_win, text="Browse", command=dcp_newdir)
    dir_button.grid(row=2, column=2, padx=20)

    boolv = BooleanVar()
    cb = Checkbutton(bar_win, text='Force use jpg (will save as png)?', variable = boolv)
    cb.grid(row=3,column=2, padx=5)
    go_button = Button(bar_win, text="Go!", command = lambda: hentAI_detection(dcp_dir=d_entry.get(), in_path=o_entry.get(), is_mosaic=False, is_video=False, force_jpg=boolv.get()))
    go_button.grid(row=3, column=1, pady=10)
    back_button = Button(bar_win, text="Back", command = backMain)
    back_button.grid(row=3,column=0, padx=10)

    bar_win.mainloop()

def mosaic_detect():
    mos_win = new_window()
    mos_win.title('Mosaic Detection')

    # input image directory label, entry, and button
    o_label = Label(mos_win, text = 'Your own input image folder: ')
    o_label.grid(row=1, padx=20 ,pady=10)
    o_entry = Entry(mos_win, textvariable=ovar)
    o_entry.grid(row=1, column=1)
    out_button = Button(mos_win, text="Browse", command=input_newdir)
    out_button.grid(row=1, column=2)

    # Entry for DCP installation
    d_label = Label(mos_win, text = 'DeepCreamPy install folder (usually called dist1): ')
    d_label.grid(row=2, padx=20, pady=20)
    d_entry = Entry(mos_win, textvariable = dvar)
    d_entry.grid(row=2, column=1, padx=20)
    dir_button = Button(mos_win, text="Browse", command=dcp_newdir)
    dir_button.grid(row=2, column=2, padx=20)

    boolv = BooleanVar()
    cb = Checkbutton(mos_win, text='Force use jpg (will save as png)?', variable = boolv)
    cb.grid(row=3,column=2, padx=5)
    go_button = Button(mos_win, text="Go!", command = lambda: hentAI_detection(dcp_dir=d_entry.get(), in_path=o_entry.get(), is_mosaic=True, is_video=False, force_jpg=boolv.get()))
    go_button.grid(row=3,column=1, pady=10)
    back_button = Button(mos_win, text="Back", command = backMain)
    back_button.grid(row=3,column=0, padx=10)


    mos_win.mainloop()

def mosaic_detect_TGAN():
    mos_win = new_window()
    mos_win.title('TecoGAN Mosaic Full decensor')

    # input image directory label, entry, and button
    o_label = Label(mos_win, text = 'Your own input image folder: ')
    o_label.grid(row=1, padx=20 ,pady=10)
    o_entry = Entry(mos_win, textvariable=ovar)
    o_entry.grid(row=1, column=1)
    out_button = Button(mos_win, text="Browse", command=input_newdir)
    out_button.grid(row=1, column=2)

    go_button = Button(mos_win, text="Go!", command = lambda: hentAI_TGAN(in_path=o_entry.get(), is_video=False))
    go_button.grid(row=2,column=1, pady=10)
    back_button = Button(mos_win, text="Back", command = backMain)
    back_button.grid(row=2,column=0, padx=10)


    mos_win.mainloop()

def video_detect_TGAN():
    mos_win = new_window()
    mos_win.title('TecoGAN Video Full decensor')

    # input image directory label, entry, and button
    o_label = Label(mos_win, text = 'Your own input video (.mp4) folder: ')
    o_label.grid(row=1, padx=20 ,pady=10)
    o_entry = Entry(mos_win, textvariable=ovar)
    o_entry.grid(row=1, column=1, padx=10)
    out_button = Button(mos_win, text="Browse", command=input_newdir)
    out_button.grid(row=1, column=2)

    go_button = Button(mos_win, text="Go!", command = lambda: hentAI_TGAN(in_path=o_entry.get(), is_video=True))
    go_button.grid(row=2,column=1, pady=10)
    back_button = Button(mos_win, text="Back", command = backMain)
    back_button.grid(row=2,column=0, padx=10)

def video_detect():
    vid_win = new_window()
    vid_win.title('Video Detection (Experimental)')

    # input video(s) directory label, entry, and button
    o_label = Label(vid_win, text = 'Your input video folder: ')
    o_label.grid(row=1, padx=20 ,pady=10)
    o_entry = Entry(vid_win, textvariable=ovar)
    o_entry.grid(row=1, column=1)
    out_button = Button(vid_win, text="Browse", command=input_newdir)
    out_button.grid(row=1, column=2)

    # Entry for DCP installation
    d_label = Label(vid_win, text = 'DeepCreamPy install folder (usually called dist1): ')
    d_label.grid(row=2, padx=20, pady=20)
    d_entry = Entry(vid_win, textvariable = dvar)
    d_entry.grid(row=2, column=1, padx=20)
    dir_button = Button(vid_win, text="Browse", command=dcp_newdir)
    dir_button.grid(row=2, column=2, padx=20)

    go_button = Button(vid_win, text="Begin Detection!", command = lambda: hentAI_detection(dcp_dir=d_entry.get(), in_path=o_entry.get(), is_mosaic=True, is_video=True))
    go_button.grid(row=3, columnspan=2, pady=5)

    vid_label = Label(vid_win, text= 'If you finished the video uncensoring, make images from DCP output back into video format. Check README for usage.')
    vid_label.grid(row=4, pady=5, padx=4)
    vid_button = Button(vid_win, text='Begin Video Maker!', command = lambda: hentAI_video_create(dcp_dir=d_entry.get(), video_path=o_entry.get()))
    vid_button.grid(row=5, pady=5, padx=10, column=1)
    back_button = Button(vid_win, text="Back", command = backMain)
    back_button.grid(row=5, padx=10, column=0)

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

# This main funct to fall back on
def backMain():
    title_window = new_window()
    title_window.title("hentAI v." + versionNumber)

    # dvar = StringVar(root)
    # ovar = StringVar(root)

    intro_label = Label(title_window, text='Welcome to hentAI! Please select a censor type: ')
    intro_label.pack(pady=10)
    bar_button = Button(title_window, text="Bar", command=bar_detect)
    bar_button.pack(pady=10)
    mosaic_button = Button(title_window, text="Mosaic", command=mosaic_detect)
    mosaic_button.pack(pady=10)
    mosaic_TG_button = Button(title_window, text="Mosaic (TecoGAN)", command=mosaic_detect_TGAN)
    mosaic_TG_button.pack(pady=10)
    video_button = Button(title_window, text='Video (Experimental)', command=video_detect)
    video_button.pack(pady=10, padx=10)

    title_window.geometry("300x200")
    title_window.mainloop()

if __name__ == "__main__":
    title_window = new_window()
    title_window.title("hentAI v." + versionNumber)

    dvar = StringVar(root)
    ovar = StringVar(root)

    intro_label = Label(title_window, text='Welcome to hentAI! Please select a censor type: ')
    intro_label.pack(pady=10)
    bar_button = Button(title_window, text="Bar", command=bar_detect)
    bar_button.pack(pady=10)
    mosaic_button = Button(title_window, text="Mosaic (DCP)", command=mosaic_detect)
    mosaic_button.pack(pady=10)
    mosaic_TG_button = Button(title_window, text="Mosaic (TecoGAN)", command=mosaic_detect_TGAN)
    mosaic_TG_button.pack(pady=10)
    video_button = Button(title_window, text='Video (DCP)', command=video_detect)
    video_button.pack(pady=10, padx=10)
    video_TG_button = Button(title_window, text="Video (TecoGAN)", command=video_detect_TGAN) # separate window for future functionality changes
    video_TG_button.pack(pady=10, padx=10)
    detect_instance = Detector(weights_path=weights_path)
    detect_instance.load_weights()
    title_window.geometry("300x300")
    title_window.mainloop()

    pass