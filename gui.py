import pathlib, os
import customtkinter as tk
import numpy as np
from customtkinter import filedialog
from PIL import Image
import random
import glob

fake_frames, real_frames = [], []
fake_frames_idx, real_frames_idx = 0, 0
generator_model_file_path, generator_class_file_path = "", ""

class App(tk.CTk): #App inherits from custom tkinter library
    def __init__(self): #Constructor method
        super().__init__() #Calls the parent class' constructor: tk.CTk
        print("Initialised app")

        self.title("Video Frame Interpolation") #Window title
        
        #Sets window size automatically based on screen size
        width = self.winfo_screenwidth()               
        height = self.winfo_screenheight() 
        self.geometry("%dx%d" % (width, height))
        #Sets default appearance as dark
        tk.set_appearance_mode("Dark")

        self.grid_columnconfigure(1, weight=4)
        self.grid_rowconfigure(0, weight=1)
        print("Loading the GUI")

        #Calls functions to add tk items to the GUI
        self.createLeftSidebar()
        self.createImageFrame()
        self.createRightSidebar()
        
        #Sets initial values on sliders droplists etc.
        self.appearanceMenu.set("Dark")

    def createLeftSidebar(self):
        #Makes frame to contain buttons, labels etc.
        self.leftSidebarFrame = tk.CTkFrame(self)
        self.leftSidebarFrame.grid(row=0, column=0, rowspan=4, padx=(20,0), pady=(20, 10), sticky="nsew")
        self.leftSidebarFrame.grid_rowconfigure(5, weight=1)

        #Buttons here are for everything except the compression
        self.buttonExampleLoad = tk.CTkButton(master=self.leftSidebarFrame, text="Interpolation Example", command=self.example_frames) #Load initial image
        self.buttonExampleLoad.grid(row=0, column=0, padx=20, pady=(20, 10)) #Sets position inside of the frame

        self.buttonImageReload = tk.CTkButton(master=self.leftSidebarFrame, text="Load Generator Model", command=self.load_generator) #Reload image without compressions
        self.buttonImageReload.grid(row=1, column=0, padx=20, pady=(20, 10))

        self.buttonImageReload = tk.CTkButton(master=self.leftSidebarFrame, text="Generate Frames", ) #Reload image without compressions
        self.buttonImageReload.grid(row=2, column=0, padx=20, pady=(20, 10))

        self.appearanceLabel = tk.CTkLabel(master=self.leftSidebarFrame, text="Appearance Mode:") #Change GUI theme
        self.appearanceLabel.grid(row=6, column=0, padx=20, pady=(10, 0))

        self.appearanceMenu = tk.CTkOptionMenu(self.leftSidebarFrame, values=["System", "Light", "Dark"], command=self.appearanceChange)
        self.appearanceMenu.grid(row=7, column=0, padx=20, pady=(10, 10))

        self.scalingLabel = tk.CTkLabel(master=self.leftSidebarFrame, text="UI Scaling:") #Change scale
        self.scalingLabel.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = tk.CTkOptionMenu(self.leftSidebarFrame, values=["100%", "90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%"], command=self.scalingChange)
        self.scaling_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 20))

    def createImageFrame(self):
        #Stores image in middle frame
        self.labelImageLoad = tk.CTkFrame(self, corner_radius=0)
        self.labelImageLoad.grid(row=0, column=1, rowspan=4, columnspan=2, padx=(20, 0), pady=(0, 5), sticky="nsew")
        self.labelImageLoad.grid_columnconfigure(0, weight=1)
        self.labelImageLoad.grid_columnconfigure(1, weight=1)
        self.labelImageLoad.grid_rowconfigure(1, weight=1)

        self.leftimagetext = tk.CTkLabel(master=self.labelImageLoad, text="")
        self.leftimagetext.grid(row=0, column=0, pady=(50, 0))

        self.rightimagetext = tk.CTkLabel(master=self.labelImageLoad, text="")
        self.rightimagetext.grid(row=0, column=1, pady=(50, 0))

        self.buttonFrameBackward = tk.CTkButton(master=self.labelImageLoad, text="<", command=self.backwardFrames) #Load initial image
        self.buttonFrameBackward.grid(row=2, column=0, padx=20, pady=(20, 10)) #Sets position inside of the frame
        self.buttonFrameForward = tk.CTkButton(master=self.labelImageLoad, text=">", command=self.forwardFrames) #Load initial image
        self.buttonFrameForward.grid(row=2, column=1, padx=20, pady=(20, 10)) #Sets position inside of the frame

        self.textConsoleLog = tk.CTkTextbox(master=self.labelImageLoad, width=900, corner_radius=1, text_color="white", border_color="#333", border_width=2)
        self.textConsoleLog.grid(row=3, column=0, columnspan=2)

    def createRightSidebar(self):
        #Creates tabber for compression buttons and sliders + statistics split into sections
        self.rightSidebarFrame = tk.CTkFrame(self)
        self.rightSidebarFrame.grid(row=0, column=5, rowspan=6, padx=(20,0), pady=(20, 10), sticky="nsew")
        self.rightSidebarFrame.grid_rowconfigure(5, weight=1)
        self.ganTitle = tk.CTkLabel(self.rightSidebarFrame, text="Design a GAN Model:")
        self.ganTitle.grid(row=0, column=5)
        self.tabview = tk.CTkTabview(self.rightSidebarFrame)
        self.tabview.grid(row=1, column=5, rowspan=4, padx=20, pady=(5, 10), sticky="nsew")
        self.tabview.add("Generator")
        self.tabview.add("Discriminator")
        self.tabview.add("Train")
        self.tabview.add("Evaluations")
        self.tabview.tab("Generator").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Discriminator").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Train").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Evaluations").grid_columnconfigure(0, weight=1)

        self.buttonFrameForward = tk.CTkButton(master=self.rightSidebarFrame, text="Begin Training", command=self.beginTraining) #Load initial image
        self.buttonFrameForward.grid(row=6, column=5, padx=20, pady=(20, 10)) #Sets position inside of the frame

    def example_frames(self):
        global fake_frames, real_frames
        fake_frames = glob.glob('src/vimeo_septuplet/sequences/00016/0016/*.png')
        real_frames = [fake_frames[0], fake_frames[2], fake_frames[4], fake_frames[6]]
        self.leftimage = tk.CTkImage(light_image=Image.open(real_frames[0]), size=(448, 256))
        self.leftimagelabel = tk.CTkLabel(self.labelImageLoad, image=self.leftimage, text="")
        self.leftimagelabel.grid(row=1, column=0)
        self.rightimage = tk.CTkImage(light_image=Image.open(fake_frames[0]), size=(448, 256))
        self.rightimagelabel = tk.CTkLabel(self.labelImageLoad, image=self.rightimage, text="")
        self.rightimagelabel.grid(row=1, column=1)

        self.leftimagetext.configure(text=f"Limited frames ({real_frames_idx+1}/{len(real_frames)})")
        self.rightimagetext.configure(text=f"Interpolated frames ({fake_frames_idx+1}/{len(fake_frames)})")

    def backwardFrames(self):
        global fake_frames_idx, real_frames_idx
        if fake_frames_idx == 0 and real_frames_idx == 0:
            fake_len = len(fake_frames)
            real_len = len(real_frames)
            self.leftimagelabel.configure(image=tk.CTkImage(light_image=Image.open(real_frames[real_len-1]), size=(448, 256)))
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=Image.open(fake_frames[fake_len-1]), size=(448, 256)))
            fake_frames_idx = fake_len - 1
            real_frames_idx = real_len - 1
        elif fake_frames_idx != 0 and real_frames_idx == 0:
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=Image.open(fake_frames[fake_frames_idx-1]), size=(448, 256)))
            fake_frames_idx = fake_frames_idx - 1
        elif fake_frames_idx != 0 and real_frames_idx != 0:
            self.leftimagelabel.configure(image=tk.CTkImage(light_image=Image.open(real_frames[real_frames_idx-1]), size=(448, 256)))
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=Image.open(fake_frames[fake_frames_idx-1]), size=(448, 256)))
            fake_frames_idx = fake_frames_idx - 1
            real_frames_idx = real_frames_idx - 1

        self.leftimagetext.configure(text=f"Limited frames ({real_frames_idx+1}/{len(real_frames)})")
        self.rightimagetext.configure(text=f"Interpolated frames ({fake_frames_idx+1}/{len(fake_frames)})")
        

    def forwardFrames(self):
        global fake_frames_idx, real_frames_idx
        if fake_frames_idx == len(fake_frames) - 1 and real_frames_idx == len(real_frames) - 1:
            self.leftimagelabel.configure(image=tk.CTkImage(light_image=Image.open(real_frames[0]), size=(448, 256)))
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=Image.open(fake_frames[0]), size=(448, 256)))
            fake_frames_idx = 0
            real_frames_idx = 0
        elif fake_frames_idx != len(fake_frames) - 1 and real_frames_idx == len(real_frames) - 1:
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=Image.open(fake_frames[fake_frames_idx+1]), size=(448, 256)))
            fake_frames_idx = fake_frames_idx + 1
        elif fake_frames_idx != len(fake_frames) - 1 and real_frames_idx != len(real_frames) - 1:
            self.leftimagelabel.configure(image=tk.CTkImage(light_image=Image.open(real_frames[real_frames_idx+1]), size=(448, 256)))
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=Image.open(fake_frames[fake_frames_idx+1]), size=(448, 256)))
            fake_frames_idx = fake_frames_idx + 1
            real_frames_idx = real_frames_idx + 1

        self.leftimagetext.configure(text=f"Limited frames ({real_frames_idx+1}/{len(real_frames)})")
        self.rightimagetext.configure(text=f"Interpolated frames ({fake_frames_idx+1}/{len(fake_frames)})")
        
    def beginTraining(self):
        pass

    def appearanceChange(self, new: str):
        tk.set_appearance_mode(new) #Updates GUI theme

    def scalingChange(self, new: str):
        newFloat = int(new.replace("%", "")) / 100
        tk.set_widget_scaling(newFloat) #Updates GUI scaling

    def load_generator(self):
        global generator_class_file_path, generator_model_file_path
        generator_model_file_path = filedialog.askopenfilename(title="Select a generator model", filetypes=[("PTH files", "*.pth")])
        generator_class_file_path = filedialog.askopenfilename(title="Select a python script", filetypes=[("Python files", "*.py")])

if __name__ == "__main__":
    app = App() #Creates new app instance
    app.mainloop() #Keeps app running by using a loop