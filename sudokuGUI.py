import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from PIL import Image, ImageTk
from detection import *

# Used to store path of image
filepath=None


def onClose():

    '''
    Callback function closed when window is closed

    '''

    if messagebox.askokcancel("Quit","Do you want to Quit?"):
        root.destroy()
        exit()


def showImage(img):

    '''
    Used to show image

    Parameters:
        img (numpy.ndarray) : Image to be displayed
    
    '''

    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()


def solveSudoku():

    '''
    Solves Sudoku
    
    '''

    res, resultImg=sudokuDetection(filepath)
    if res:
        showImage(resultImg)


def addImage():

    '''
    Function used to add image to window
    
    '''

    try:
        img=Image.open(filepath)
        img=img.resize((300,300),Image.ANTIALIAS)
        img=ImageTk.PhotoImage(img)
        image_label=tk.Label(root, image=img)
        image_label.image=img
        image_label.pack()
        solveBtn=tk.Button(root,text="Solve",bd="2",command=solveSudoku)
        solveBtn.pack()
    except Exception as e:
        print(e)

def addText():

    '''
    Function used to add text to window

    '''

    text=tk.Text(master=root,height=10,width=30)
    text.pack()
    text.insert(tk.END,filepath)


def chooseFile():

    '''
    Displays a file chooser

    '''

    global filepath
    tk.Tk().withdraw()
    name=askopenfilename()
    filepath=name
    addImage()


# Root element of window
root=tk.Tk()
root.geometry("500x500")
root.title("Sudoku Detection")
label=tk.Label(root, text="Sudoku Detection").pack()
btn=tk.Button(root,text="Choose Image",bd="2",command=chooseFile)
btn.pack(side="top")
root.protocol("WM_DELETE_WINDOW",onClose)
root.mainloop()