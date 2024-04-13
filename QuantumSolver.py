import subprocess
import sys
import re
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from scipy.constants import m_e, hbar


REPLACEMENTS = {
    r'cos\((.*?)\)': r'np.cos(\1)',
    r'sin\((.*?)\)': r'np.sin(\1)',
    r'tan\((.*?)\)': r'np.tan(\1)',
    r'cosh\((.*?)\)': r'np.cosh(\1)',
    r'sinh\((.*?)\)': r'np.sinh(\1)',
    r'tanh\((.*?)\)': r'np.tanh(\1)',
    r'cosh\^(\d+)\((.*?)\)': r'np.power(np.cosh(\2), \1)',
    r'sinh\^(\d+)\((.*?)\)': r'np.power(np.sinh(\2), \1)',
    r'tanh\^(\d+)\((.*?)\)': r'np.power(np.tanh(\2), \1)',
    r'\((.*?)\)\*\*(\d+)': r'np.power(\1, \2)',
    r'x\*\*(\d+)': r'np.power(x, \1)',
    r'H\((.*?)\)': r'np.heaviside(\1, 0.5)'
}

EVAL_DICT = {'x': 0, 'np': np, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 
            'cosh': np.cosh, 'sinh': np.sinh, 'tanh': np.tanh, 'acos': np.arccos, 
            'asin': np.arcsin, 'atan': np.arctan, 'acosh': np.arccosh, 'asinh': np.arcsinh, 
            'atanh': np.arctanh, 'H': np.heaviside}



class QuantumSolver:
    def __init__(self):
        super().__init__()
        #Setting the tolerance for the wave fonction at the ending point (x_max) to accept the energy level as the wnated energy level 7-0
        self.tolerance = 0.0001
        #Setting the initial augmentation after the point where the wave function will be set to zero 4-0
        self.initial_augmentation = 0.001
        #Setting the number of division from the initial point in the classical forbidden zone x_0 to the ending point x_max
        self.nbr_division = 2000
        self.nbr_division_V = 2000
        
        # Initialize all the parameters as class attributes
        self.y_max = None
        self.min_x = None
        self.max_x = None
        self.WavPlot = None
        self.WavLines = None
        self.EnergyLines = None
        self.PositionPotential = None
        self.PotentialArray = None
        

    def replace_absolute_values(self, potential):
        pot_list = potential.rsplit('|')
        for i in [ i for i in range(1,(len(pot_list)-1)*2) if i%2==1 ]:
            insertion = 'np.absolute(' if i%4 ==1 else ')'
            pot_list.insert(i,insertion)
        return ''.join(pot_list)

    def modify_potential(self, potential):
        """This function replaces any mathematical expression that is usually used but that is incorrect in python."""
        for pattern, replacement in REPLACEMENTS.items():
            potential = re.sub(pattern, replacement, potential)
        potential = potential.replace('^','**')
        potential = self.replace_absolute_values(potential)
        return potential

    def verify_and_correct_potential(self, potential):
        while True:
            #Tries to evaluate the potential at x=0 and asks for a new one until there is no more syntax error
            try:
                x=0
                # Check if potential is a string and does not contain an integer followed by parentheses
                if isinstance(potential, str) and not re.search(r'\d+\(\)', potential):
                    eval(potential, EVAL_DICT)
                    break
                else:
                    raise SyntaxError
            except SyntaxError:
                try:
                    pot_msg = CTkMessagebox(title="Warning Message!",
                                    message="Input error. Please try again.", 
                                    icon="warning", option_1="Cancel", option_2="Retry")
                    if pot_msg.get()=="Retry":
                        text = "Write the new potential!"
                        action = ctk.CTkInputDialog(text=text, title="New Potential")
                        potential = action    
                except EOFError:
                    msg = CTkMessagebox(title="Warning Message!", 
                            message="Input error. Please try again.", icon="warning", option_1="Cancel", option_2="Retry")
                    if msg.get()=="Retry":
                        continue
                potential = self.modify_potential(potential)

        return potential


    def verify_limits_potential(self, potential):

        #Verify if the potential is bigger than V(x=0) for x=100 and x=-100
        
        while True:
            eval_pot = [eval(potential, {**EVAL_DICT, 'x': x, 'np': np, 'H': np.heaviside}) for x in [-100, 100]]
            if any(i < eval(potential, {**EVAL_DICT, 'x': 0, 'np': np, 'H': np.heaviside}) for i in eval_pot):
                try:
                    msg = CTkMessagebox(title="Info", message='The potential doesn\'t seem to be correct. Do you want to correspond it to a bound state?',
                                        icon="question", option_1="Cancel", option_2="No", option_3="Yes")
                    response = msg.get()
                except EOFError:
                    msg = CTkMessagebox(title="Warning Message!", 
                            message="Input error. Please try again.", icon="warning", option_1="Cancel", option_2="Retry")
                    if msg.get()=="Retry":
                        continue
                if response == "Yes":
                    try:
                        text = "Write the new potential!"
                        action = ctk.CTkInputDialog(text=text, title="New Potential")
                        potential = action
                    except EOFError:
                        msg = CTkMessagebox(title="Warning Message!", 
                                message="Input error. Please try again.", icon="warning", option_1="Cancel", option_2="Retry")
                        if msg.get()=="Retry":
                            continue
                    #Check the syntax for the new potential
                    potential = self.modify_potential(potential)
                    potential = self.verify_and_correct_potential(potential)
                else:
                    break

            #If it respects the condition, exit the while loop
            else :
                break

        return potential

    def calculate_initial_energy_guess(self, potential_array):
        min_potential = potential_array.min()
        mean_potential = potential_array.mean()
        initial_energy_guess = min_potential +  (1/500000) * (mean_potential + min_potential)
        return initial_energy_guess


    def verify_potential_concavity(self, potential_array, initial_energy_guess):
        """
        This function verifies the concavity of a potential energy curve and adjusts the initial energy guess if necessary.

        Args:
            potential_array (numpy.ndarray): An array of potential energy values.
            initial_energy_guess (float): An initial guess for the energy.

        Raises:
            ValueError: If the energy guess reaches zero and the function is unable to verify the potential concavity.

        Returns:
            tuple: A tuple containing the concavity of the potential ('positive' or 'negative') and the adjusted initial energy guess.
        """

        while True:
            index_min = []
            index_max = []

            # Tries to find meeting points and to compare them
            for i in range(len(potential_array) - 1):
                # Gets all the points where the potential meets the E_verify value and filters them depending on their derivatives
                if potential_array[i] > initial_energy_guess and potential_array[i+1] < initial_energy_guess:
                    index_min.append(i)
                elif potential_array[i] < initial_energy_guess and potential_array[i+1] > initial_energy_guess:
                    index_max.append(i)
                elif potential_array[i] == initial_energy_guess:
                    if potential_array[i-1] > initial_energy_guess and potential_array[i+1] < initial_energy_guess:
                        index_min.append(i)
                    elif potential_array[i-1] < initial_energy_guess and potential_array[i+1] > initial_energy_guess:
                        index_max.append(i)

            # Defines the concavity value depending on the meeting points
            if (max(index_max) > max(index_min)) and (min(index_max) > min(index_min)):
                concavity = 'positive'
            else:
                concavity = 'negative'

            # If we are not able to compare the potential, we define a new energy guess
            if not index_min or not index_max:
                initial_energy_guess /= 2
                if initial_energy_guess == 0:
                    raise ValueError("Energy guess has reached zero. Unable to verify potential concavity.")
            else:
                break

        return concavity, initial_energy_guess

    def evaluate_potential_at_position(self, position,potential):
        
        x = position
        evaluated_potential = eval(potential)

        return evaluated_potential

    def centering_potential(self, position_potential, potential_array):

        # Get the minimum value for the potential and the translation in y
        translation_y = potential_array.min()

        # Translate the potential
        potential_array -=  translation_y

        return position_potential, potential_array

    def center_potential(self, potential,trans_y):
        '''Modify the potential expression to center its minimum at x=0 and y=0'''

        #y translation
        potential = f"({potential}) - {trans_y}"

        return potential

    def guess_energy(self, energy_levels_found, previous_energy_guesses, iteration, first_energy_guess):
        """
        This function generates a guess for the energy level of a quantum system. 
        It uses the energy levels that have already been found and the previous energy guesses to make an informed guess.

        Args:
            energy_levels_found (dict): A dictionary where the keys are the energy levels that have already been found.
            previous_energy_guesses (dict): A dictionary of previous energy guesses and their corresponding results.
            iteration (int): The current iteration number.
            first_energy_guess (float): The first guess for the energy level.

        Returns:
            float: The guessed energy level.
        """

        #If it is the first time, return the first energy level of the quantum harmonic oscillator
        if iteration == 1:
            return first_energy_guess

        Lvl_found = list(energy_levels_found.keys())
        Lvl_found.sort()
        #Gets the energy level that we want to find
        E_level_missing = [index for index,Energy in enumerate(Lvl_found) if not Energy <= index]
        if not E_level_missing:
            if not Lvl_found:
                E_level_guess = 0
            else:
                E_level_guess = max(Lvl_found) +1
        else:
            E_level_guess = min(E_level_missing)

        try:
            E_level_smaller = max([ E for E in previous_energy_guesses.keys() if E <= E_level_guess ])
        except ValueError:
            E_level_smaller = None
        try:
            E_level_bigger = min([ E for E in previous_energy_guesses.keys() if E > E_level_guess ])
        except ValueError:
            E_level_bigger = None

        #Define the energy guess
        #If the smaller and higher exist take the average
        E_guess = 0  # Initialize E_guess with a default value
        
        if (not E_level_smaller == None) and (not E_level_bigger ==None):
            E_guess = ( previous_energy_guesses[E_level_smaller][1] + previous_energy_guesses[E_level_bigger][0] ) / 2

        #If only the higher exists take the half
        elif not E_level_bigger == None:
            E_guess = previous_energy_guesses[E_level_bigger][0]/2

        #If only the smaller exists take the double
        elif not E_level_smaller == None:
            E_guess = previous_energy_guesses[E_level_smaller][1] * 2

        return E_guess

    def find_meeting_points(self, E_guess, potential_array, position_potential, previous_energy_guesses):
        """
        This function finds the points where the potential energy curve intersects a given energy level (E_guess). 
        If less than two intersection points are found, it adjusts the energy level and tries again, up to a maximum of ten attempts.

        Args:
            E_guess (float): The initial guess for the energy level.
            potential_array (numpy.ndarray): An array of potential energy values.
            position_potential (numpy.ndarray): An array of positions corresponding to the potential_array.
            previous_energy_guesses (dict): A dictionary of previous energy guesses and their corresponding results.

        Returns:
            tuple: A tuple containing the positions of the two intersection points (or None if not found), a boolean indicating whether the maximum number of attempts has been reached, and the adjusted energy guess.
        """
        #Initializing constant for the while loop
        iteration = 0
        meeting_points = (None, None)  # Replace None with actual values
        end_program = False

        while True:
            #Finds all the meeting points
            meeting_points = [None,None]
            for i in range(0,len(potential_array)-2):
                #Gets all the meeting points
                if (potential_array[i] < E_guess and potential_array[i+1] > E_guess) or (potential_array[i] > E_guess and potential_array[i+1] < E_guess) or (potential_array[i] == E_guess):
                    #And filter them
                    if (meeting_points[0] == None) or (position_potential[i] < meeting_points[0]):
                        meeting_points[0] = position_potential[i]
                    elif (meeting_points[1] == None) or (position_potential[i] > meeting_points[1]):
                        meeting_points[1] = position_potential[i]

            #If we have not found at least two meeting points, then make a new smaller energy guess and repeat for at most ten times
            if (meeting_points[0] == None) or (meeting_points[1] == None):
                E_guess = (E_guess + max([k for j,k in previous_energy_guesses.values() if k < E_guess]))/2
                iteration += 1
                if iteration > 10:
                    end_program = True
                    break
            else:
                meeting_points = tuple(meeting_points)
                break

        return meeting_points,end_program,E_guess


    def determine_min_and_max(self, meeting_points):

        #Sets the min and max as the half of the distance between the min and the max plus the min or the max
        position_min = meeting_points[0] - (meeting_points[1] - meeting_points[0])/1
        position_max =  meeting_points[1] + (meeting_points[1] - meeting_points[0])/1

        return position_min,position_max


    # Calculate the wave function
    def wave_function_numerov(self, potential_func, E_guess, nbr_division, initial_augmentation, position_min, position_max):

        # Calculate division and its square once
        division = (position_max - position_min) / nbr_division
        division_squared = division ** 2

        # Convert potential_func to a function if it's a string
        if isinstance(potential_func, str):
            potential_func = eval("lambda x: " + potential_func)

        # Initialize wave_function and position_array
        index = 0
        wave_function = [(float(position_min), 0), (float(position_min + division), initial_augmentation)]
        position_array = np.arange(position_min, position_max, division)

        # Calculate wave function for other values
        for i in np.arange(position_min + (2 * division), position_max, division):
            # Evaluate the potential
            V_plus1 = potential_func(i)
            V = potential_func(position_array[index + 1])
            V_minus1 = potential_func(position_array[index])

            #Setting the k**2 values ( where k**2 = (2m/HBar^2)*(E-V(x)) )
            k_2_plus1 = 2 * (E_guess- V_plus1)
            k_2 = 2 * (E_guess - V)
            k_2_minus1 = 2 * (E_guess - V_minus1)

            # Calculate the wave function
            psi = ((2 * (1 - (5/12) * division_squared * k_2) * wave_function[-1][1]) - (1 + (1/12) * division_squared * k_2_minus1) * wave_function[-2][1]) / (1 + (1/12) * division_squared * k_2_plus1)

            # Save the wave function and the x coordinate
            wave_function.append((i, psi))

            #Incrementing the index
            index += 1

        return wave_function

    # Determine the number of nodes in the wave function
    def number_of_nodes(self, wave_function):

        #Initialize the number of nodes and their position
        NumberOfNodes = 0
        PositionNodes = list()

        #Calculate the number of nodes
        for i in range(1,len(wave_function)-1):
            if (wave_function[i][1] > 0 and wave_function[i+1][1] < 0) or (wave_function[i][1] < 0 and wave_function[i+1][1] > 0) or (wave_function[i][1] == 0):
                NumberOfNodes += 1
                PositionNodes.append(wave_function[i][0])


        #Gets the biggest position
        x = list()
        for position,wave in wave_function:
            x.append(position)
        x_max = max(x)

        return NumberOfNodes,PositionNodes,x_max

    # Verify if wave function respects the restriction

    def verify_tolerance(self, wave_function, tolerance, E_guess, E_guess_try, number_of_nodes):

        # Check if the last value of the wave function respects the tolerance
        verification_tolerance = np.absolute(wave_function[-1][1]) < tolerance 

        # Check if the energy guess doesn't change a lot
        try:
            E_minus = E_guess_try[number_of_nodes][1]
            E_plus = E_guess_try[number_of_nodes + 1][0]
        except KeyError:
            pass
        else:
            if (E_guess < E_plus and E_guess > E_minus) and ((E_minus/E_plus) > 0.9999999999) :
                verification_tolerance = True

        return verification_tolerance

    def correct_number_of_nodes(self, number_of_nodes, position_nodes, x_max, E_guess, E_guess_try):

        number_of_corrected_nodes = number_of_nodes
        #Correct the number of nodes if E_guess is between the lowest energy for this number of nodes and the maximum for the number of nodes - 1
        try:
            if (E_guess_try[number_of_nodes][1] > E_guess) and (E_guess_try[number_of_nodes - 1][1] < E_guess):
                number_of_corrected_nodes -= 1
        #If the dictionnary E_guess_try doesn't contain these keys check if the Last number of nodes is close to the maximum value in x x_max
        except KeyError:
            if (position_nodes/x_max) > 94:
                number_of_corrected_nodes -= 1

        return number_of_corrected_nodes


    # Save energy and the correponding number of nodes

    def save_energy(self, number_of_nodes, E_guess, E_guess_try):

        #Checks if the key Number of Nodes exists. If it doesn't, define the two values in the list corresponding to the key NumberOfNodes as E_guess.
        try:
            E_guess_try[number_of_nodes]

        except KeyError:
            E_guess_try[number_of_nodes] = [E_guess, E_guess]
            return E_guess_try

        #Checks if the energy guess is smaller than the smallest value in the list
        if E_guess < E_guess_try[number_of_nodes][0]:
            E_guess_try[number_of_nodes][0] = E_guess

        #Checks if the energy guess is greater than the biggest value in the list
        elif E_guess > E_guess_try[number_of_nodes][1]:
            E_guess_try[number_of_nodes][1] = E_guess

        return E_guess_try

    #Draw the figure
    #Define the wave funcions to plot, the lines corresponding to these wave function and the energy lines
    def define_what_to_plot(self, WaveFunctionFound, EnergyLevelFound):

        # Determine the maximum energy to set the maximum value for the y axis
        y_max = 1.1*EnergyLevelFound[max(EnergyLevelFound)]
        Y_by_E_level = (y_max/(max(EnergyLevelFound)+2))

        # For the wave function
        WavPlot = []
        x = np.array([])  # Initialize x
        for i in WaveFunctionFound.keys():
            x_temp=[]
            y=[]
            for j in range(400,len(WaveFunctionFound[i])-240):
                if not (j > 3750 and np.absolute(WaveFunctionFound[i][j][1]) > (max(y)*0.07)):
                    x_temp.append(WaveFunctionFound[i][j][0])
                    y.append(WaveFunctionFound[i][j][1])
            x_temp = np.array(x_temp)
            y = np.array(y)

            mult = (0.9 * Y_by_E_level)/(2 * y.max())
            y = (mult * y) + (Y_by_E_level * (i+1))
            WavPlot.append((x_temp,y))
            x = np.concatenate((x, x_temp))  # Update x with new values

        # Determines the min and max in x
        min_x = x.min()
        max_x = x.max()

        # Get lines to where the wave function is centered
        WavLines = []
        for i in WaveFunctionFound.keys():
            Wav_line_y=[]
            for j in range(len(x)):
                Wav_line_y.append(Y_by_E_level * (i+1))
            WavLines.append((x,Wav_line_y))

        # Get lines for all the Energy levels
        EnergyLines = []
        for i in WaveFunctionFound.keys():
            En_y = []
            for j in range(len(x)):
                En_y.append(EnergyLevelFound[i])
            EnergyLines.append((x,En_y))

        return y_max, min_x, max_x, WavPlot, WavLines, EnergyLines

    def calculate_parameters(self, WaveFunctionFound, EnergyLevelFound):
        # Call the DefineWhatToPlot method to calculate the parameters
        self.y_max, self.min_x, self.max_x, self.WavPlot, self.WavLines, self.EnergyLines = self.define_what_to_plot(WaveFunctionFound, EnergyLevelFound)

    def draw_wave_functions(self, Wav):
        lines = [Wav.plot(x,y,'b',label=r"$Re(\psi(x))$",zorder=3)[0] for x,y in self.WavPlot]
        lines2 =  [Wav.plot(x,y,'m',label=r"$Im(\psi(x))$",zorder=3)[0] for x,y in self.WavPlot]

        for x,y in self.WavLines:
            Wav.plot(x,y,'k--',zorder=1)

        Wav.axis([self.min_x, self.max_x, 0, self.y_max])
        Wav.plot(self.PositionPotential, self.PotentialArray, 'r',label='Potential',zorder=2)

        return lines, lines2

    def draw_energy_levels(self, En):
        i = 0
        for x,y in self.EnergyLines:
            PlotColor = cm.viridis(i/len(self.EnergyLines)) # type: ignore
            En.plot(x,y,'--',color=PlotColor,label='E'+str(i),zorder=2)
            i+=1

        En.axis([self.min_x, self.max_x, 0, self.y_max])
        En.plot(self.PositionPotential, self.PotentialArray, 'r',label='Potential',zorder=1)

    def set_wave_function_plot(self, Wav):
        Wav.set_xlabel(r'x ($\AA$)')
        Wav.set_title('Wave Function',fontsize=14)

        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)
        leg1 = Wav.legend(newHandles, newLabels, loc='upper left', fontsize='x-small')
        leg1.get_frame().set_alpha(1)

        for i in range(len(self.EnergyLines)):
            Wav.text(((self.max_x - self.min_x) * 0.04) + self.min_x, self.WavLines[i][1][0] - (0.25 * (self.y_max/(len(self.EnergyLines)+2))), r'$\Psi_{%s}(x)$'%(i))

    def set_energy_levels_plot(self, En):
        En.set_xlabel(r'x ($\AA$)')
        En.set_ylabel('Energy')
        En.set_title('Energy levels',fontsize=14)
        leg2 = En.legend(loc='upper left', fontsize='x-small')
        leg2.get_frame().set_alpha(1)

    def animate_wave_function(self, f, lines, lines2):
        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def UpdateData(t):
            for j,line in enumerate(lines):
                x = self.WavPlot[j][0]
                y = ((self.WavPlot[j][1] - (self.WavLines[j][1][0]))  * np.cos(self.EnergyLines[j][1][0]*t/20)) + (self.WavLines[j][1][0])
                line.set_data(x,y)
            for j,line in enumerate(lines2):
                x = self.WavPlot[j][0]
                y = ((self.WavPlot[j][1] - (self.WavLines[j][1][0]))  * np.sin(self.EnergyLines[j][1][0]*t/20)) + (self.WavLines[j][1][0])
                line.set_data(x,y)

            return lines,lines2

        anim = animation.FuncAnimation(f, UpdateData, init_func=init, interval=10, blit=False, repeat=True, save_count=300, ) # type: ignore

        return anim

    def on_plot_render(self, widget):
        # Destroy the messagebox here
        widget.destroy()

    def plotter(self, messagebox):
        f,(En,Wav) = plt.subplots(1,2,sharey=True)
        f.suptitle("Schrödinger equation solutions",fontsize=20,fontweight='bold')

        lines, lines2 = self.draw_wave_functions(Wav)
        self.draw_energy_levels(En)
        
        self.set_wave_function_plot(Wav)
        self.set_energy_levels_plot(En)

        anim = self.animate_wave_function(f, lines, lines2)

        fig = plt.gcf()
        fig.set_size_inches(9.25, 5.25, forward=True)


        # Connect to the 'draw_event'
        fig.canvas.mpl_connect('draw_event', lambda event: self.on_plot_render(messagebox))

        plt.show()
        return anim


    def get_user_input(self,values):
        self.x_V_min = float(values[0])
        self.x_V_max = float(values[1])
        self.E_level = int(values[2])
        self.potential = values[3]

    def process_potential(self):
        self.potential = self.modify_potential(self.potential)
        self.potential = self.verify_and_correct_potential(self.potential)
        self.potential = self.verify_limits_potential(self.potential)

    def calculate_potential_array(self):
        EvaluatePotential = np.vectorize(self.evaluate_potential_at_position)
        DivisionPotential = (self.x_V_max - self.x_V_min) / self.nbr_division_V
        PositionPotential = np.arange(self.x_V_min, self.x_V_max, DivisionPotential)
        return EvaluatePotential(PositionPotential, self.potential)

    def adjust_potential_array(self, PotentialArray, PositionPotential):
        PositionPotential, PotentialArray = self.centering_potential(PositionPotential, PotentialArray)
        #PotentialArray, PositionPotential = self.center_potential(PotentialArray, PositionPotential)
        return PotentialArray, PositionPotential

    def check_potential(self,values):
        self.get_user_input(values)

        while True:
            self.process_potential()

            PotentialArray = self.calculate_potential_array()
            DivisionPotential = (self.x_V_max - self.x_V_min) / self.nbr_division_V
            PositionPotential = np.arange(self.x_V_min, self.x_V_max, DivisionPotential)
            PotentialArray, PositionPotential = self.adjust_potential_array(PotentialArray, PositionPotential)

            First_E_guess = self.calculate_initial_energy_guess(PotentialArray)
            concavity, First_E_guess = self.verify_potential_concavity(PotentialArray, First_E_guess)

            if concavity == 'positive':
                break
            elif concavity == 'negative':
                msg = CTkMessagebox(title="Info", message='The potential you entered does not have the correct concavity.If you want to override this warning and use the current potential anyway click yes',
                                        icon="question", option_1="Cancel", option_2="No", option_3="Yes")
                response = msg.get()
                if response == "Yes":
                    break
                else:
                    text = "Write the new potential!"
                    action = ctk.CTkInputDialog(text=text, title="New Potential")
                    potential2 = action
                    self.potential = potential2
        # Assign PotentialArray and PositionPotential to instance variables
        self.PotentialArray = PotentialArray
        self.PositionPotential = PositionPotential
        return First_E_guess, self.PotentialArray, self.PositionPotential, self.potential
        
    def run(self, values):
        #Initializing paramaters for the while loop
        EnergyLevelFound = {} 
        WaveFunctionFound = {} 
        E_guess_try = {} 
        iteration = 1 
        E_found = list() 
        First_E_guess,PotentialArray, PositionPotential, potential = self.check_potential(values)
        while not E_found == list(range(0,self.E_level)):
            # Initial Energy guess
            E_guess = self.guess_energy(EnergyLevelFound, E_guess_try, 
                                        iteration, First_E_guess)
            # Setting the initial and final points (where \psi=0)
            # Gets the meeting points with the energy and the potential
            MeetingPoints, end_program, E_guess = self.find_meeting_points(E_guess, PotentialArray, 
                                                                        PositionPotential, E_guess_try)

            if end_program:
                break

            # Sets the minimum and maximum value for the position where the wave function equals zero
            Position_min, Position_max = self.determine_min_and_max(MeetingPoints)

            # Calculate the wave fonction for the guessed energy value
            WaveFunction = self.wave_function_numerov(potential, E_guess, self.nbr_division, 
                                                    self.initial_augmentation, Position_min, Position_max)
            NumberOfNodes,PositionNodes,x_max = self.number_of_nodes(WaveFunction)
            VerificationTolerance = self.verify_tolerance(WaveFunction, self.tolerance, E_guess,
                                                        E_guess_try, NumberOfNodes)
            if VerificationTolerance:
                NumberOfNodesCorrected = self.correct_number_of_nodes(NumberOfNodes, PositionNodes,
                                                                    x_max, E_guess, E_guess_try)
                EnergyLevelFound.update({NumberOfNodesCorrected:E_guess})
                WaveFunctionFound.update({NumberOfNodesCorrected:WaveFunction})

            # Save Energy guess and the corresponding number of nodes (no matter if it fails)
            E_guess_try = self.save_energy(NumberOfNodes, E_guess, E_guess_try)
            iteration += 1
            # Update the Energy levels found list to verify if the condition is respected
            E_found = list()
            for i in EnergyLevelFound.keys():
                E_found.append(i)
            E_found.sort()

        return WaveFunctionFound, EnergyLevelFound
    
    def plot(self,values,messagebox):
        WaveFunctionFound, EnergyLevelFound = self.run(values)
        self.calculate_parameters(WaveFunctionFound, EnergyLevelFound)
        #Draw all the wave functions
        self.plotter(messagebox)
