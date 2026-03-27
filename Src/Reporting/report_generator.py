'''
Created on 18 juil. 2023

@author: aletard
'''

#----------------------------------------------------------------#
#                                                                #
#                    External imports                            #
#                                                                #
#----------------------------------------------------------------#

import sys
import matplotlib.pyplot as plt
import os


#----------------------------------------------------------------#
#                                                                #
#                    Packages imports                            #
#                                                                #
#----------------------------------------------------------------#

from Src.utils.repository_manager import RepositoryManager as RM



#----------------------------------------------------------------#
#                                                                #
#                    Global variables                            #
#                                                                #
#----------------------------------------------------------------#




#----------------------------------------------------------------#
#                                                                #
#                    Abstract Classes                            #
#                                                                #
#----------------------------------------------------------------#




#----------------------------------------------------------------#
#                                                                #
#                    Functions & Classes                         #
#                                                                #
#----------------------------------------------------------------#

class ReportGenerator():
   
    def __init__(self, output_repository, simulator_config):
        '''
        Constructor
        '''
        self.output_repositiory_path = output_repository
                
        RM.create_repository(f"{self.output_repositiory_path}/logs" )
        self.logs_path = RM.get_absolute_from_relative_path(f"{self.output_repositiory_path}/logs/logs.txt")
        
        RM.create_repository(f"{self.output_repositiory_path}/results" )
        self.results_path = RM.get_absolute_from_relative_path(f"{self.output_repositiory_path}/results")
        
        RM.create_repository(f"{self.output_repositiory_path}/config")
        self.config_report(RM.get_absolute_from_relative_path(f"{self.output_repositiory_path}/config/config.txt"), simulator_config)

      
        #-----------------------
 
        
    def log_generator(self, message):

        # Console display for quick notice
        print(message)
    
    
        # if file exist, write following the last log, otherwise create the file
        try:
            with open(self.logs_path, "a", encoding='utf-8') as logs:
                sys.stdout = logs
                print(message)
                # Go back to original outpout
                sys.stdout = sys.__stdout__
        except:
            with open(self.logs_path, "w", encoding='utf-8') as logs:
    
                sys.stdout = logs
                print(message)
                # Go back to original outpout
                sys.stdout = sys.__stdout__

        # -------------------------------------------------------------------
        
    def config_report(self, config_path, simulator_config ):

        message = f"Simulation configuration: \n" + \
                    f"Dataset: {simulator_config[0]}, {simulator_config[1]} iterations, algorithm: {simulator_config[2]}"

        # Console display for quick notice
        print(message)
    
    
        # if file exist, write following the last log, otherwise create the file
        try:
            with open(config_path, "a", encoding='utf-8') as config:
                sys.stdout = config
                print(message)
                # Go back to original outpout
                sys.stdout = sys.__stdout__
        except:
            with open(config_path, "w", encoding='utf-8') as config:
    
                sys.stdout = config
                print(message)
                # Go back to original outpout
                sys.stdout = sys.__stdout__

        # -------------------------------------------------------------------

    def save_accuracy_plot(self, iterations, accuracy, filename="accuracy.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, accuracy, color='darkgreen', linestyle='-', 
             linewidth=1.5, marker='.', markersize=2)        
        plt.xlabel("Nombre d'itérations")
        plt.ylabel("Accuracy")
        plt.title("Évolution de l'accuracy")

        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        
        save_path = os.path.join(self.results_path, filename)
        
        plt.savefig(save_path)
        plt.close()

        # -------------------------------------------------------------------

    def save_cpc_plot(self, iterations, cpc, filename="cpc.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, cpc, color='darkgreen', linestyle='-', 
             linewidth=1.5, marker='.', markersize=2) 
        plt.xlabel("Nombre d'itérations")
        plt.ylabel("CPC")
        plt.title("Évolution du CPC")
        
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        
        save_path = os.path.join(self.results_path, filename)
        
        plt.savefig(save_path)
        plt.close()

        # -------------------------------------------------------------------
