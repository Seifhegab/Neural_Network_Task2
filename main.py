import tkinter as tk
import pandas as pd
from tkinter import *
from Preprocess import Preprocess
from Encoding_Data import EncodingData
from Splitting_Data import SplittingData
from Multi_Layer_Preceptron import MultiLayerPerceptron


class LayersTable:
    numOfLayers = 0
    layersData = []
    entries = []

    def __init__(self, numOfLayers, frame):
        self.numOfLayers = numOfLayers
        self.frame = frame
        self.saveBtn = tk.Button(self.frame, text="Save", command=self.saveData)

    def render(self):
        rows = 0
        for i in range(self.numOfLayers):
            # Check if added 4 columns, increase the number of rows
            if i % 8 == 0 and i != 0:
                rows = rows + 1

            layerEntry = tk.Entry(self.frame, width=8)
            layerEntry.grid(row=rows, column=i % 8, padx=5, pady=5)
            self.entries.append(layerEntry)
        self.saveBtn.grid()

    def saveData(self):
        # Getting value of every entry and push it into array
        for entry in self.entries:
            numOfNeurons = int(entry.get())
            self.layersData.append(numOfNeurons)
        return self.layersData


def renderHandler():
    numOfLayers = int(numOfLayersEntry.get())
    layers = LayersTable(numOfLayers, layersFrame)
    layers.render()


def Change_Type():
    # number of layers
    numOfLayers = int(numOfLayersEntry.get()) + 1
    # number of neurons in each layer
    layers = LayersTable(numOfLayers, layersFrame)
    Neurons_list = layers.layersData
    Neurons_list.append(3)
    # learning rate
    LearningRate = float(LearningRate_text.get("1.0", "end-1c"))
    # number of epochs
    Epochs = int(Epochs_text.get("1.0", "end-1c"))
    # bias check
    biasCheck = CheckVar1.get()
    # Activation
    Activation = RadioVar1.get()
    # training and test data
    x_train, y_train, x_test, y_test = Calling_Func()

    c = MultiLayerPerceptron(x_train,y_train,x_test,y_test,LearningRate,biasCheck,Epochs,Neurons_list,numOfLayers,
                             Activation)
    c.generateLayers()


def Calling_Func():
    x = pd.read_csv("Dry_Bean_Dataset.csv").iloc[:, 0:5]
    y = pd.read_csv("Dry_Bean_Dataset.csv").iloc[:, 5:6]
    pre = Preprocess(x)
    pre.fill_null()
    x = pre.Normalize()
    encode = EncodingData(y)
    y = encode.label_encode()
    splitting = SplittingData(x, y)
    x_train, y_train, x_test, y_test = splitting.split()
    return x_train, y_train, x_test, y_test

m = tk.Tk()
m.title('Main Frame')
m.geometry("1000x1000")

numOfLayersLabel = tk.Label(m, text="Number Of Hidden Layers")
numOfLayersLabel.pack()
numOfLayersLabel.place(x=0, y=0)
numOfLayersEntry = tk.Entry(m)
numOfLayersEntry.pack()
numOfLayersEntry.place(x=150, y=0)

renderBtn = tk.Button(m, text="Render", command=renderHandler)
renderBtn.pack()
renderBtn.place(x=300, y=0)

layersFrame = tk.Frame(m)
layersFrame.pack(side=tk.TOP, padx=10, pady=40)

LearningRate_label = tk.Label(m, text='Please enter the learning rate')
LearningRate_label.pack()
LearningRate_label.place(x=0, y=400)
LearningRate_text = tk.Text(m, height=1, width=20)
LearningRate_text.pack()
LearningRate_text.place(x=200, y=400)

Epochs_label = tk.Label(m, text='Please enter the number of epochs')
Epochs_label.pack()
Epochs_label.place(x=0, y=450)
Epochs_text = tk.Text(m, height=1, width=20)
Epochs_text.pack()
Epochs_text.place(x=200, y=450)


Bias_label = tk.Label(m, text='Please choose if you want bias or not: ')
Bias_label.pack()
Bias_label.place(x=0, y=550)
CheckVar1 = IntVar()
Bias_Checkbox = tk.Checkbutton(m, text="Bias", activebackground="black", activeforeground="white", bd=0,
                               variable=CheckVar1,
                               onvalue=1, offvalue=0)
Bias_Checkbox.pack()
Bias_Checkbox.place(x=210, y=550)

Perceptron_label = tk.Label(m, text='Please choose your Activation Function: ')
Perceptron_label.pack()
Perceptron_label.place(x=0, y=600)
RadioVar1 = IntVar()
Perceptron_radiobutton = tk.Radiobutton(m, text="Sigmoid", variable=RadioVar1, value=1)
Perceptron_radiobutton.pack()
Perceptron_radiobutton.place(x=210, y=600)
adaline_radiobutton = tk.Radiobutton(m, text="Hyperbolic Tangent", variable=RadioVar1, value=2)
adaline_radiobutton.pack()
adaline_radiobutton.place(x=210, y=650)

generate_btn = tk.Button(m, text='Generate', width=15, command=Change_Type)
generate_btn.pack()
generate_btn.place(x=500, y=700)

# Calling_Func()

m.mainloop()
