'''
Created on 23 mars 2024

@author: aletard
'''


#----------------------------------------------------------------#
#                                                                #
#                    External imports                            #
#                                                                #
#----------------------------------------------------------------#


import numpy as np


#----------------------------------------------------------------#
#                                                                #
#                    Packages imports                            #
#                                                                #
#----------------------------------------------------------------#




#----------------------------------------------------------------#
#                                                                #
#                    Global variables                            #
#                                                                #
#----------------------------------------------------------------#


# ... No global variables defined ...


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


class ResultStorer():
    
    def __init__(self, horizon):

        self.start_time = None
        self.end_time = None
        self.simulation_duration = None
        
        self.threshold = 4
        self.algorithm_performance ={"predicted_arms": np.empty(horizon, dtype=object),
                                     "correctness" : np.zeros(horizon),
                                     "accuracy" : np.zeros(horizon),
                                     "cumulated_regrets" : np.zeros(horizon),
                                     "cpc" : np.zeros(horizon)
                                     }
            

        #-----------------------

    def update_measures(self, iteration, observed_value, price=0, rewards=0):

        self.update_correctness(iteration, observed_value)
        self.update_accuracy(iteration)
        self.update_regrets(iteration)
        self.update_cpc(iteration, price, rewards)
        
        #-----------------------

    def update_correctness(self, iteration, observed_value):
            
            chosen_arms = self.algorithm_performance["predicted_arms"][iteration]
            
            # Si aucun bras n'a été choisi
            if chosen_arms is None or (isinstance(chosen_arms, list) and len(chosen_arms) == 0):
                self.algorithm_performance["correctness"][iteration] = 0
                return
                
            if not isinstance(chosen_arms, list):
                chosen_arms = [chosen_arms]
                
            good_arms_count = 0
             
            for arm in chosen_arms:
                feedback_rows = observed_value["feedback"][observed_value["arm_id"] == arm]
                
                if len(feedback_rows) > 0:
                    feedback = feedback_rows.iloc[0]
                    if feedback >= self.threshold:
                        good_arms_count += 1
                        
            self.algorithm_performance["correctness"][iteration] = good_arms_count / len(chosen_arms)
            #-----------------------
        
    def update_accuracy(self, iteration):
        self.algorithm_performance["accuracy"][iteration] = \
                np.sum(self.algorithm_performance["correctness"]) / (iteration + 1)
 
 
        #-----------------------
        
    def update_regrets(self, iteration):
        
        if iteration == 0 :
            self.algorithm_performance["cumulated_regrets"][iteration] = 1 - self.algorithm_performance["correctness"][0]
        else: 
            self.algorithm_performance["cumulated_regrets"][iteration] = \
                self.algorithm_performance["cumulated_regrets"][iteration-1] + (1 - self.algorithm_performance["correctness"][iteration])
                             
                                                 
    #-----------------------------------------------------------------------------------------

    def update_cpc(self, iteration, price, rewards): 
        if rewards == 0:
            current_cpc = 0.0 
        else:
            current_cpc = price / rewards  
        self.algorithm_performance["cpc"][iteration] = current_cpc
     


