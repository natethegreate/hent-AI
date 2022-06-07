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
import configparser
import shutil
from detector import Detector


versionNumber = '1.6.9'
weights_path = 'weights.h5' # should call it weights.h5 in main dir
cfg_path = 'hconfig.ini'

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

def hentAI_detection(dcp_dir=None, in_path=None, is_mosaic=False, is_video=False, force_jpg=False, dilation=0):
    # TODO: Create new window? Can show loading bar
    # hent_win = new_window()
    # info_label = Label(hent_win, text="Beginning detection")
    # info_label.pack(padx=10,pady=10)
    # hent_win.mainloop()

    if dcp_dir==None:
        error(5)
    if in_path==None:
        error(2)

    # Update config with new vars
    hconfig = configparser.ConfigParser()
    hconfig.read(cfg_path)
    if 'USER' in hconfig:
        hconfig['USER']['dcpdir'] = dcp_dir
        hconfig['USER']['srcdir'] = in_path
        hconfig['USER']['gmask'] = str(dilation)
    else:
        print("ERROR in hentAI_detection: Unable to read config file")
    with open(cfg_path, 'w') as cfgfile:
        hconfig.write(cfgfile)

    dilation = (dilation) * 2 # Dilation performed via kernel, so dimension is doubled
          
    if(is_mosaic == True and is_video==False):
        # Copy input folder to decensor_input_original. NAMES MUST MATCH for DCP
        print('copying inputs into input_original dcp folder')
        # print(in_path)
        # print(listdir(in_path))
        for fil in listdir(in_path):
            if fil.endswith('jpg') or fil.endswith('png') or fil.endswith('jpeg') or fil.endswith('JPG') or fil.endswith('PNG') or fil.endswith('JPEG'):
                try:
                    shutil.copy(in_path + '/' + fil, dcp_dir + '/decensor_input_original/' + fil) # DCP is compatible with original jpg input.
                except Exception as e:
                    print("ERROR in hentAI_detection: Mosaic copy to decensor_input_original failed!", fil, e)
                    return

    # Run detection
    if(is_video==True):
        print('Running video detection')
        loader = Tk()
        loader.title('Running detections')
        load_label = Label(loader, text='Now running detections. This can take around a minute or so per image. Please wait')
        load_label.pack(side=TOP, fill=X, pady=10, padx=20)
        loader.update()
        detect_instance.run_on_folder(input_folder=in_path, output_folder=dcp_dir+'/decensor_input/', is_video=True, orig_video_folder=dcp_dir + '/decensor_input_original/', dilation=dilation) #no jpg for video detect
        loader.destroy()
    else:
        print('Running detection, outputting to dcp input')
        loader = Tk()
        loader.title('Running detections')
        load_label = Label(loader, text='Now running detections. This can take around a minute or so per image. Please wait')
        load_label.pack(side=TOP, fill=X, pady=10, padx=20)
        loader.update()
        detect_instance.run_on_folder(input_folder=in_path, output_folder=dcp_dir+'/decensor_input/', is_video=False, is_mosaic=is_mosaic, dilation=dilation)
        loader.destroy()


    # Announce completion, TODO: offer to run DCP from DCP directory
    detect_instance.unload_model()
    print('Process complete!')
    popup = Tk()
    popup.title('Success!')
    label = Label(popup, text='Process executed successfully! Now you can run DeepCreamPy.')
    label.pack(side=TOP, fill=X, pady=20, padx=10)
    num_jpgs = detect_instance.get_non_png()
    # dcprun = Button(popup, text='Run DCP (Only if you have the .exe)', command= lambda: run_dcp(dcp_dir))
    # dcprun.pack(pady=10)
    okbutton = Button(popup, text='Ok', command=popup.destroy)
    okbutton.pack()
    popup.mainloop()

# helper function to call TGAN folder function. 
def hentAI_TGAN(in_path=None, is_video=False, force_jpg=True):
    print("Starting ESRGAN detection and decensor")
    
    # Update config with new vars
    hconfig = configparser.ConfigParser()
    hconfig.read(cfg_path)
    if 'USER' in hconfig:
        hconfig['USER']['srcdir'] = in_path
    else:
        print("ERROR in hentAI_detection: Unable to read config file")
    with open(cfg_path, 'w') as cfgfile:
        hconfig.write(cfgfile)
    loader = Tk()
    loader.title('Running ESRGAN')
    load_label = Label(loader, text='Now running decensor. This can take a while. Please wait')
    load_label.pack(side=TOP, fill=X, pady=10, padx=20)
    loader.update()
    detect_instance.run_ESRGAN(in_path = in_path, is_video = is_video, force_jpg = force_jpg)
    loader.destroy()

    print('Process complete!')
    popup = Tk()
    popup.title('Success!')
    label = Label(popup, text='Process executed successfully!')
    label.pack(side=TOP, fill=X, pady=20, padx=10)
    num_jpgs = detect_instance.get_non_png()

    okbutton = Button(popup, text='Ok', command=popup.destroy)
    okbutton.pack()
    popup.mainloop()

# globals that hold directory strings
dtext = ""
otext = ""
mtext = ""

# Use hconfig.ini to populate the directory strings
def get_cfg():
    hconfig = configparser.ConfigParser()
    hconfig.read(cfg_path)
    if 'USER' in hconfig:
        dtext = hconfig['USER']['dcpdir']
        otext = hconfig['USER']['srcdir']
        mtext = hconfig['USER']['gmask']
    else:
        print("ERROR in get_cfg: Unable to read USER section")
    
    dvar.set(dtext)
    ovar.set(otext)
    mvar.set(mtext)



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
    d_label = Label(bar_win, text = 'DeepCreamPy install folder: ')
    d_label.grid(row=2, padx=20, pady=10)
    d_entry = Entry(bar_win, textvariable = dvar)
    d_entry.grid(row=2, column=1, padx=20)
    dir_button = Button(bar_win, text="Browse", command=dcp_newdir)
    dir_button.grid(row=2, column=2, padx=20)

    dil_label = Label(bar_win, text='Grow detected mask amount (0 to 10)')
    dil_label.grid(row=3, padx=10, pady=10)
    dil_entry = Entry(bar_win, textvariable = mvar)
    dil_entry.grid(row=3, column=1, padx=20)

    go_button = Button(bar_win, text="Go!", command = lambda: hentAI_detection(dcp_dir=d_entry.get(), in_path=o_entry.get(), is_mosaic=False, is_video=False, force_jpg=True, dilation=int(dil_entry.get()) ))
    go_button.grid(row=4, column=1, pady=10)
    back_button = Button(bar_win, text="Back", command = backMain)
    back_button.grid(row=4,column=0, padx=10)

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
    d_label = Label(mos_win, text = 'DeepCreamPy install folder: ')
    d_label.grid(row=2, padx=20, pady=20)
    d_entry = Entry(mos_win, textvariable = dvar)
    d_entry.grid(row=2, column=1, padx=20)
    dir_button = Button(mos_win, text="Browse", command=dcp_newdir)
    dir_button.grid(row=2, column=2, padx=20)

    dil_label = Label(mos_win, text='Grow detected mask amount (0 to 10)')
    dil_label.grid(row=3, padx=10, pady=10)
    dil_entry = Entry(mos_win, textvariable = mvar)
    dil_entry.grid(row=3, column=1, padx=20)

    # boolv = BooleanVar()
    # cb = Checkbutton(mos_win, text='Force use jpg (will save as png)?', variable = boolv)
    # cb.grid(row=3,column=2, padx=5) # Removing Force jpg option because jpg always works
    go_button = Button(mos_win, text="Go!", command = lambda: hentAI_detection(dcp_dir=d_entry.get(), in_path=o_entry.get(), is_mosaic=True, is_video=False,dilation=int(dil_entry.get()), force_jpg=True))
    go_button.grid(row=4,column=1, pady=10)
    back_button = Button(mos_win, text="Back", command = backMain)
    back_button.grid(row=4,column=0, padx=10)


    mos_win.mainloop()

def mosaic_detect_TGAN():
    mos_win = new_window()
    mos_win.title('ESRGAN Mosaic Full decensoring')

    # input image directory label, entry, and button
    o_label = Label(mos_win, text = 'Your own input image folder: ')
    o_label.grid(row=1, padx=20 ,pady=10)
    o_entry = Entry(mos_win, textvariable=ovar)
    o_entry.grid(row=1, column=1)
    out_button = Button(mos_win, text="Browse", command=input_newdir)
    out_button.grid(row=1, column=2, padx=10)
    help_label = Label(mos_win, text = 'Output can be found in ESR_output/ folder')
    help_label.grid(row=2, column=1, padx=10)

    go_button = Button(mos_win, text="Go!", command = lambda: hentAI_TGAN(in_path=o_entry.get(), is_video=False))
    go_button.grid(row=3,column=1, pady=10)
    back_button = Button(mos_win, text="Back", command = backMain)
    back_button.grid(row=3,column=0, padx=10)


    mos_win.mainloop()

def video_detect_TGAN():
    mos_win = new_window()
    mos_win.title('ESRGAN Video Full decensor (Nvidia GPU Highly reccommended)')

    # input image directory label, entry, and button
    o_label = Label(mos_win, text = 'Your own input video (.mp4) folder: ')
    o_label.grid(row=1, padx=20 ,pady=10)
    o_entry = Entry(mos_win, textvariable=ovar)
    o_entry.grid(row=1, column=1, padx=10)
    out_button = Button(mos_win, text="Browse", command=input_newdir)
    out_button.grid(row=1, column=2, padx=10)

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
    d_label = Label(vid_win, text = 'DeepCreamPy install folder: ')
    d_label.grid(row=2, padx=20, pady=20)
    d_entry = Entry(vid_win, textvariable = dvar)
    d_entry.grid(row=2, column=1, padx=20)
    dir_button = Button(vid_win, text="Browse", command=dcp_newdir)
    dir_button.grid(row=2, column=2, padx=20)

    dil_label = Label(vid_win, text='Grow detected mask amount (0 to 10)')
    dil_label.grid(row=3, padx=10, pady=10)
    dil_entry = Entry(vid_win, textvariable = mvar)
    dil_entry.grid(row=3, column=1, padx=20)

    go_button = Button(vid_win, text="Begin Detection!", command = lambda: hentAI_detection(dcp_dir=d_entry.get(), in_path=o_entry.get(), is_mosaic=True,dilation=int(dil_entry.get()), is_video=True))
    go_button.grid(row=4, columnspan=2, pady=5)

    vid_label = Label(vid_win, text= 'When you finish the DCP uncensoring, Video Maker packs images from DCP output back into video format. Check the tutorial for more, output in main directory.')
    vid_label.grid(row=5, pady=5, padx=4)
    vid_button = Button(vid_win, text='Begin Video Maker!', command = lambda: hentAI_video_create(dcp_dir=d_entry.get(), video_path=o_entry.get()))
    vid_button.grid(row=6, pady=5, padx=10, column=1)
    back_button = Button(vid_win, text="Back", command = backMain)
    back_button.grid(row=6, padx=10, column=0)

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

# This main funct to fall back on, if the back button is pressed
def backMain():
    title_window = new_window()
    title_window.title("hentAI v." + versionNumber)

    # Clear string entry vars, but repopulate from config
    dvar.set('')
    ovar.set('')
    mvar.set('')
    get_cfg()

    intro_label = Label(title_window, text='Welcome to hentAI! Please select a censor type: ')
    intro_label.pack(pady=10)
    bar_button = Button(title_window, text="Bar", command=bar_detect)
    bar_button.pack(pady=10)
    mosaic_button = Button(title_window, text="Mosaic (DCP)", command=mosaic_detect)
    mosaic_button.pack(pady=10)
    mosaic_TG_button = Button(title_window, text="Mosaic (ESRGAN)", command=mosaic_detect_TGAN)
    mosaic_TG_button.pack(pady=10)
    video_button = Button(title_window, text='Video (DCP)', command=video_detect)
    video_button.pack(pady=10, padx=10)
    video_TG_button = Button(title_window, text="Video (ESRGAN)", command=video_detect_TGAN) # separate window for future functionality changes
    video_TG_button.pack(pady=10, padx=10)
    title_window.geometry("300x300")
    title_window.mainloop()

if __name__ == "__main__":
    title_window = new_window()
    title_window.title("hentAI v." + versionNumber)

    # apparently global variables for text entries
    dvar = StringVar(root)
    ovar = StringVar(root)
    mvar = StringVar(root)

    intro_label = Label(title_window, text='Welcome to hentAI! Please select a censor type: ')
    intro_label.pack(pady=10)
    bar_button = Button(title_window, text="Bar", command=bar_detect)
    bar_button.pack(pady=10)
    mosaic_button = Button(title_window, text="Mosaic (DCP)", command=mosaic_detect)
    mosaic_button.pack(pady=10)
    mosaic_TG_button = Button(title_window, text="Mosaic (ESRGAN)", command=mosaic_detect_TGAN)
    mosaic_TG_button.pack(pady=10)
    video_button = Button(title_window, text='Video (DCP)', command=video_detect)
    video_button.pack(pady=10, padx=10)
    video_TG_button = Button(title_window, text="Video (ESRGAN)", command=video_detect_TGAN) # separate window for future functionality changes
    video_TG_button.pack(pady=10, padx=10)
    detect_instance = Detector(weights_path=weights_path)
    get_cfg()
    # detect_instance.load_weights() # instance will load weights on its own
    title_window.geometry("300x300")
    title_window.mainloop()

    pass