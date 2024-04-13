import os
import subprocess
import sys
import textwrap
import threading
import queue

import customtkinter as ctk

from PIL import Image
from CTkMessagebox import CTkMessagebox
from QuantumSolver import QuantumSolver

class GUI:
    def __init__(self):
        super().__init__()
        self.last_pressed_button = None  
        self.current_frame = None
################## Widgets Initialization ###############   

        # Create the main window
        self.root = ctk.CTk()
        self.width, self.height = 1000, 560

        # set grid layout 1x2
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.geometry(f"{self.width}x{self.height}+400+200")

        self.root.resizable(False,False)
        self.root.iconbitmap(os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/logo.ico"))
        self.root.title('Arithmetic Solution of Schrödinger Equation')        

        # Adding Images to GUI
        img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
        self.logo_image = ctk.CTkImage(Image.open(os.path.join(img_path, "Shrodinger.jpg")), size=(26, 26))
        self.finite_well_image = ctk.CTkImage(Image.open(os.path.join(img_path, "Finite Well.png")), size=(20, 20))
        self.harmonic_oscillator_image = ctk.CTkImage(Image.open(os.path.join(img_path, "Harmonic_oscillator.png")), size=(20, 20))
        self.Poeschl_Teller_image = ctk.CTkImage(Image.open(os.path.join(img_path, "Poeschl-Teller-potential.png")), size=(20, 20))
        self.double_well_image = ctk.CTkImage(Image.open(os.path.join(img_path, "DoubleWell.png")), size=(20, 20))
        self.multiple_well_image = ctk.CTkImage(Image.open(os.path.join(img_path, "Multiple-Well-Potential.png")), size=(20, 20))

        # create navigation frame
        self.parent_frame = ctk.CTkFrame(self.root, corner_radius=0)
        self.parent_frame.grid(row=0, column=0, sticky="nsew")
        self.parent_frame.grid_rowconfigure(6, weight=1)
        
        
        self.parent_frame_label = ctk.CTkLabel(self.parent_frame, text="  Arithmetic Solution of Schrödinger Equation", image=self.logo_image,
                                                    compound="left", font=ctk.CTkFont(size=15, weight="bold"))
        self.parent_frame_label.grid(row=0, column=0, padx=20, pady=20)

        # Create the navigation frame Buttons
        self.finite_well = self.create_button("Finite Well", 1, 0, self.finite_well_image, 
                                            lambda event: self.button_event("Finite Well"))
        self.harmonic_oscillator = self.create_button("Harmonic Oscillator", 2, 0, self.harmonic_oscillator_image, 
                                            lambda event: self.button_event("Harmonic Oscillator"))
        self.Poeschl_Teller = self.create_button("Pöschl - Teller Potential", 3, 0, self.Poeschl_Teller_image, 
                                            lambda event: self.button_event("Pöschl - Teller Potential"))
        self.double_well = self.create_button("Double Well", 4, 0, self.double_well_image, 
                                            lambda event: self.button_event("Double Well"))
        self.multiple_well = self.create_button("Multiple Well - Hypergrid", 5, 0, self.multiple_well_image, 
                                            lambda event: self.button_event("Multiple Well - Hypergrid"))
        
        # create first frame
        self.first_frame = ctk.CTkFrame(self.root, corner_radius=10, fg_color="transparent")
        self.create_widgets(self.first_frame)
        self.text = textwrap.dedent("""\
        Write in the entry box bellow the desired potential function
        that you want to be process with Schrödinger Equation.
        The potential function that express the finite well can be written in 
        general as: V(x) = -Vo*H(a/2+x)*H(a/2-x), multiplying these Heaviside 
        step functions by -Vo ensures that the potential is -Vo within the well 
        and 0 outside the well""")
        self.insert_text_into_textbox_generic(self.information_textbox,self.text)
        

        # create second frame
        self.second_frame = ctk.CTkFrame(self.root, corner_radius=10, fg_color="transparent")
        self.create_widgets(self.second_frame)
        self.text = textwrap.dedent("""\
        Write in the entry box bellow the desired potential function
        that you want to be process with Schrödinger Equation.
        The potential function that express the harmonic oscillator can be written in 
        general as: V(x) = 1/2*k*x^2, where k: force constant of the oscillator""")
        self.insert_text_into_textbox_generic(self.information_textbox,self.text)

        # create third frame
        self.third_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        self.create_widgets(self.third_frame)
        self.text = textwrap.dedent("""\
        Write in the entry box bellow the desired potential function
        that you want to be process with Schrödinger Equation.
        The potential function that express the Pöschl - Teller can be written in 
        general as: V(x) = -λ*(λ+1)/(2*cosh^2(a*x)), where λ,a: is common constants.
        The parameter λ influences the shape and depth of the potential well, while the parameter 
        a controls the width of the well""")
        self.insert_text_into_textbox_generic(self.information_textbox,self.text)
        
        #create fourth frame
        self.fourth_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        self.create_widgets(self.fourth_frame)
        self.text = textwrap.dedent("""\
        Write in the entry box bellow the desired potential function
        that you want to be process with Schrödinger Equation.
        The potential function that express the symmetric double well can be written in 
        general as: V(x) = b*x^4-c*x^2, where b,c are constants that 
        determine the characteristic length scale.
        On the other hand the asymmetric double well can be written in 
        general as: V(x) = a*x^4-b*x^2-c*x, where a, b, c, and d are constants 
        that determine the shape of the potential energy curve.
        This potential function represents an asymmetric double-well potential. 
        The term a*x^4 is responsible for the double-well nature,as it introduces the fourth power of x.
        The term b*x^2 contributes to the curvature of each well, and the c⋅x term introduces the asymmetry.""")
        self.insert_text_into_textbox_generic(self.information_textbox,self.text)
        
        #create fifth frame
        self.fifth_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        self.create_widgets(self.fifth_frame)
        self.text = textwrap.dedent("""\
        Write in the entry box bellow the desired potential function
        that you want to be process with Schrödinger Equation.
        The potential function for a system with multiple wells can 
        vary depending on the specific configuration and properties 
        of the wells. However, a common way to model a system with 
        multiple wells is to consider a sum of individual well potentials.
        For a simple example, consider a system with two wells. The potential energy function 
        V(x) could be represented as the sum of two Pöschl-Teller potentials:
        V(x) = -λ1*(λ1+1)/(2*cosh^2(a1*x)) - λ2*(λ2+1)/(2*cosh^2(a2*x)), where each well
        is being described by it's constant values.""")
        self.insert_text_into_textbox_generic(self.information_textbox,self.text)

        # Create the appearance mode menu
        self.appearance_mode_label = ctk.CTkLabel(self.parent_frame, text="Appearance Mode:", anchor="center")
        self.appearance_mode_label.place(relx = 0.34, rely = 0.73)
        self.appearance_mode_menu = ctk.CTkOptionMenu(self.parent_frame, values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.appearance_mode_menu.set("System")
        self.appearance_mode_menu.place(relx = 0.31, rely = 0.78)


        # Create the scaling menu
        self.scaling_label = ctk.CTkLabel(self.parent_frame, text="UI Scaling:", anchor="center")
        self.scaling_label.place(relx = 0.40, rely = 0.87)
        self.scaling_optionemenu = ctk.CTkOptionMenu(self.parent_frame, values=["80%", "90%", "100%"],
                                                    command=self.change_scaling_event)
        self.scaling_optionemenu.place(relx = 0.30, rely = 0.92)
        self.scaling_optionemenu.set("100%")

        # Dictionary mapping names to buttons and frames
        self.name_to_button_frame = {
            "Finite Well": (self.finite_well, self.first_frame),
            "Harmonic Oscillator": (self.harmonic_oscillator, self.second_frame),
            "Pöschl - Teller Potential": (self.Poeschl_Teller, self.third_frame),
            "Double Well": (self.double_well, self.fourth_frame),
            "Multiple Well - Hypergrid": (self.multiple_well, self.fifth_frame)
        }
        
        # Set focus to the widgets
        self.root.focus_set()
        # Update the display
        self.root.update_idletasks() 

        # Bind the Enter key to the enter_pressed method
        self.root.bind("<Return>", lambda event: self.enter_pressed())

        # Bind the Escape key to the on_closing method
        self.root.bind("<Escape>", lambda event: self.on_closing(event))
        
        # Close the window
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        
########## Functions for the widgets ##############

    # Creates button for the parent/left frame
    def create_button(self, name, row_number, column_number, image, command=None):
        """
        Creates a button with the specified properties and adds it to the parent frame.

        Args:
            name (str): The text to display on the button.
            row_number (int): The row number where the button should be placed in the grid.
            column_number (int): The column number where the button should be placed in the grid.
            image: The image to display on the button.
            command (Callable, optional): The function to call when the button is clicked. Defaults to None.

        Returns:
            ctk.CTkButton: The created button.
        """
        button = ctk.CTkButton(
        self.parent_frame,
        corner_radius=10,
        height=40,
        border_spacing=10,
        text=name,
        fg_color="transparent",
        text_color=("gray10", "gray90"),
        hover_color=("gray70", "gray30"),
        image=image,
        anchor="w" if command else "center"
        )
        button.grid(row=row_number, column=column_number, sticky="ew")
        if command:
           button.bind("<Button-1>", command)
        return button

    def button_event(self, button_name):
        self.last_pressed_button = None

        self.current_frame = self.select_frame_by_name(button_name)
        
        # Save the new button in the variable
        self.last_pressed_button = button_name

    def select_frame_by_name(self, name):
        selected_frame = None  # Initialize selected_frame with a default value
        # Iterate over all names
        for current_name, (button, frame) in self.name_to_button_frame.items():
            if current_name == name:
                # Set button color for selected button and show frame
                button.configure(fg_color=("gray75", "gray25"))
                frame.grid(row=0, column=1, sticky="nsew")
                selected_frame = frame
            else:
                # Set button color to transparent and hide frame
                button.configure(fg_color="transparent")
                frame.grid_forget()

        return selected_frame  # Return the displayed frame
        
    
    def create_widgets(self,frame):
        # Create the textbox
        self.clear_entry_widgets(frame)
        Read_Instructions = ctk.CTkLabel(frame, text="*Read the Instructions carefully before entering a potential function!", 
                                        text_color="red", font=("default", 15, "bold"))
        
        Read_Instructions.place(relx=0.1, rely=0.87)
        self.information_textbox = ctk.CTkTextbox(frame, corner_radius=20, 
                                    width=470, height = 120, fg_color="transparent",
                                    text_color=("gray10", "gray90"), wrap='none', activate_scrollbars=True)  # type: ignore
        self.information_textbox.place(relx=0.1, rely=0.06)
        self.information_textbox.configure(state="disabled")  # Make the TextBox read-only
        
        self.minimum_x_label = ctk.CTkLabel(frame, text='Enter the minimum x value for potential evaluation -> [-100, 0):')
        self.minimum_x_label.place(relx=0.1, rely=0.3)
        self.minimum_x_entry = ctk.CTkEntry(frame, width=140, corner_radius=20)
        self.minimum_x_entry.place(relx=0.7, rely=0.3)
        self.insert_text_in_Entrybox(self.minimum_x_entry)
        self.maximum_x_label = ctk.CTkLabel(frame, text='Enter the maximum x value for potential evaluation -> (0, 100]:')
        self.maximum_x_label.place(relx=0.1, rely=0.40)
        self.maximum_x_entry = ctk.CTkEntry(frame, width=140, corner_radius=20)
        self.maximum_x_entry.place(relx=0.7, rely=0.40)
        self.insert_text_in_Entrybox(self.maximum_x_entry)
        self.energy_levels_label = ctk.CTkLabel(frame, text='Which first energy levels do you want (enter an integer):')
        self.energy_levels_label.place(relx=0.1 , rely=0.50)
        self.energy_levels_entry = ctk.CTkEntry(frame, width=140, corner_radius=20)
        self.energy_levels_entry.place(relx=0.65, rely=0.50)
        self.insert_text_in_Entrybox(self.energy_levels_entry)
        self.potential_label = ctk.CTkLabel(frame, text='Write the desired potential (as a function of x):')
        self.potential_label.place(relx=0.1, rely=0.6)
        self.potential_entry = ctk.CTkEntry(frame, width=200, corner_radius=20)
        self.potential_entry.place(relx=0.56, rely=0.6)
        self.insert_text_in_Entrybox(self.potential_entry)

        # Add enter button
        self.enter_btn = ctk.CTkButton(frame,
                                    text="Enter",
                                    font=("Arial", 12),
                                    command=self.enter_pressed)
        self.enter_btn.place(relx=0.38, rely=0.77)
        
    
    def insert_text_into_textbox_generic(self, widget_name, text):
        # Temporarily set the state to "normal"
        widget_name.configure(state="normal") 
        # Change the font style
        widget_name.configure("bold", font=('Courier',12))
        # Insert the text
        widget_name.insert("0.0", "Instructions:\n\n" + text)
        # Set the state back to "disabled"
        widget_name.configure(state="disabled")
    
    # Handle the text in EntryBoxes
    def insert_text_in_Entrybox(self, entry_name):
        entry_name.insert(0, "Type here:")
        entry_name.bind("<FocusIn>", lambda event, entry=entry_name: self.clear_entry_input(event=event, entry=entry))
        entry_name.bind("<FocusOut>", lambda event, entry=entry_name: self.insert_default_text_in_widget(event=event, text="Type here:", entry=entry))
    
    # Clear Entry inputs
    def clear_entry_input(self, event, entry=None):
        if entry is not None:
            entry.delete(0, 'end')

    # Default text in widgets
    def insert_default_text_in_widget(self, event, text, entry=None):
        if isinstance(entry, ctk.CTkEntry) and not entry.get():
            entry.insert(0, text)
    
    def enter_pressed(self, event=None):
        if self.current_frame is None:
            CTkMessagebox(title="Error", message="Please select a frame before pressing Enter!", icon="cancel")
            return
        self.reset_flags()
        entry_widgets = self.get_entry_widgets(self.current_frame)
        self.update_flags_based_on_widget(entry_widgets)  # Update flags based on all widgets
        result = self.handle_widgets(entry_widgets)
        if result is not None:
            self.messagebox = CTkMessagebox(title="Info", message="Processing...", icon="info")
            self.root.update() 
            self.result_queue = queue.Queue()
            threading.Thread(target=self.quantumsolver, args=(result, self.result_queue)).start()
            self.root.after(100, self.check_queue)
            # Update the GUI after starting the thread and scheduling the check_queue method
            self.root.update()
            
    def check_queue(self):
        try:
            result = self.result_queue.get_nowait()
        except queue.Empty:
            # If the result is not ready yet, check again after 100 ms
            self.root.after(100, self.check_queue)
        else:
            # Display the figure here using the result
            self.quantum.plot(result, self.messagebox)

            
    # Set and reset flags to its initial values
    def reset_flags(self):
        self.all_fields_are_filled_with_valid_values = True
        self.empty_field_found = False
        self.type_here_found = False
        
    # Update flags based on all widgets
    def update_flags_based_on_widget(self, widgets):
        for widget in widgets:
            if widget.get() == "":
                self.all_fields_are_filled_with_valid_values = False
                self.empty_field_found = True
            elif widget.get().startswith("Type here:"):
                self.all_fields_are_filled_with_valid_values = False
                self.type_here_found = True

    # Get all the entry widgets from the current frame
    def get_entry_widgets(self, frame):
        entry_widgets = []
        for widget in frame.winfo_children():
            if isinstance(widget, ctk.CTkEntry):
                entry_widgets.append(widget)
        return entry_widgets  

    def clear_entry_widgets(self,frame):
        # Forget the grid of the current frame
        if frame is not None:
            # Clear all the Widgets from the window
            for widget in frame.winfo_children():
                widget.place_forget()

    # Handle the widgets in the other windows
    def handle_widgets(self, entry_widgets):

        if self.type_here_found:
            CTkMessagebox(title="Error", message="No field should contain 'Type here:", icon="cancel")
        elif self.empty_field_found:
            CTkMessagebox(title="Error", message="No field should be empty!", icon="cancel")
        elif self.all_fields_are_filled_with_valid_values:
            return self.get_entry_value(entry_widgets)
        else:
            CTkMessagebox(title="Error", message="All fields must be filled with a valid value!", icon="cancel")

    # Get the inserted values from the entryboxes
    def get_entry_value(self,entry_widgets):
        values = []
        for  entry in entry_widgets:
            entry_value = entry.get()
            values.append(entry_value)
        return values

    # Call the QuantumSolver
    def quantumsolver(self, values, result_queue):
        try:
            self.quantum = QuantumSolver()
            # At the end, put the result into the queue
            result_queue.put(values)
        except Exception as e:
            # If an exception occurs, put it into the queue
            result_queue.put(e)
        
    # Method that handles the appearance mode button
    def change_appearance_mode(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)
        
    # Method for changing the scale of the window
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)        
        
    def start(self):
        self.root.mainloop()
        
    def on_closing(self, event=None):
        self.root.destroy()
        
        
if __name__ == '__main__':
    App = GUI()
    App.start()
    
    