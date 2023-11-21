import tkinter as tk
from tkinter import ttk
import customtkinter as ctk

window = tk.Tk()
window.title('Robot GUI')
window.geometry('500x300+600+100')

window.tk.call()

top_frame = ttk.Frame(window)

bottom_frame = ttk.Frame(window)
button = ctk.CTkButton(bottom_frame, text='Order to Map',fg_color='#FF0',
                       text_color='#000', hover_color='#AA0')
button2 = ctk.CTkButton(bottom_frame, text='Call Robot')

button.pack(side='left', expand=True, fill='both', padx=10)
button2.pack(side='left', expand=True, fill='both')
bottom_frame.pack(expand=True, fill='both', padx= 10, pady=20)

window.mainloop()