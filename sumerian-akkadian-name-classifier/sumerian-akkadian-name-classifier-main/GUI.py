from glob import glob
from os.path import basename
from tkinter import ACTIVE, DISABLED, Button, Label, Text, Tk
from tkinter.filedialog import askopenfilenames
from tkinter.font import NORMAL
from Classifier import classifyNames

class GUI():

    def __init__(self, master):
        self.master = master
        master.title('Sumerian-Akkaidan Classifier')
        master.geometry('500x200')
        
        self.filepaths = sorted(glob(r'./input_files/*.txt'))

        self.addTextBox()
        self.updateTextBox()

        self.runButton = Button(
            text='Run',
            height=3,
            foreground='green',
            disabledforeground='white',
            command=self.runClassifier)
        self.runButton.pack(side='right', fill='both', expand=True)
        self.runButton["state"] = DISABLED

        self.selectButton = Button(
            text='Select File',
            height=3,
            activebackground='gray',
            command=self.selectInputFiles)
        self.selectButton.pack(side='left', fill='both', expand=True)
    
    # adds one or more files to use for the classifier
    def selectInputFiles(self):
        self.filepaths = list(
                            askopenfilenames(
                                initialdir='./input_files/',
                                title='Select File(s)',
                                filetypes=(
                                    ('Text files', '*.txt*'),
                                    ('All files', '*.*'))))
        if self.filepaths:
            self.runButton["state"] = NORMAL
        self.updateTextBox()

    # create and run the classifier object
    def runClassifier(self):
        for file in self.filepaths:
            classifyNames(file)
        self.runButton["state"] = DISABLED

    # create textbox to show input files that will be used
    def addTextBox(self):       
        label = Label(text='Input Files:', anchor='w')
        label.pack(fill='both')
        self.textbox = Text(height=7)
        self.textbox.pack(fill='both', expand=True)
    
    # updates the textbox to show what files are currently selected
    def updateTextBox(self):
        self.textbox['state'] = 'normal'
        self.textbox.delete('1.0', 'end')
        for filepath in self.filepaths:
            self.textbox.insert('end', basename(filepath) + '\n')
        self.textbox['state'] = 'disabled'

root = Tk()
gui = GUI(root)
root.mainloop()
