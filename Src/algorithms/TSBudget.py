'''
Created on 23 mars 2024

@author: aletard
'''

#--------------------------------------------------------------------#
#                                                                    #
#                          external imports                          #
#                                                                    #
#--------------------------------------------------------------------#

import random
import numpy as np


#--------------------------------------------------------------------#
#                                                                    #
#                          Packages import                           #
#                                                                    #
#--------------------------------------------------------------------#




#--------------------------------------------------------------------#
#                                                                    #
#                          Global Variables                          #
#                                                                    #
#--------------------------------------------------------------------#


#--------------------------------------------------------------------#
#                                                                    #
#                         Functions & Objects                        #
#                                                                    #
#--------------------------------------------------------------------#



class TSBudget():

    def __init__(self, arms=None, context_example=None): 
        
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "TSBudget"
        self.budget = 5

        self.arms_payoff_vectors = {"cumulated_rewards" : np.zeros(len(self.ground_arms)),
                                    "tries" : np.zeros(len(self.ground_arms)),
                                    "cost" : self.ground_arms["cost"].values
                                    }
        
        self.arm_chosen = None
        # threshold used to compute rewards, actual feedback is compared to it
        # Follow the simulator metric, but this can be changed.
        self.threshold = 4
        self.price=0

        
        # -------------------------------------------------------------------

    def run(self, observed_value, user_context=None):

        self.init_choice(observed_value)
        self.arm_chosen = self.choose_action()
        
        return self.arm_chosen

        # -------------------------------------------------------------------

    def init_choice(self, observation):

        self.arm_chosen = -1
        # Ensuring algorithm only arms for which feedback have been provided by current user
        self.arms_pool = self.ground_arms[self.ground_arms["arm_id"].isin(observation["arm_id"])]
        self.arms_pool.reset_index(inplace=True)
        
        # -------------------------------------------------------------------

    def choose_action(self):

        arm_pool_size = len(self.arms_pool['arm_id']) 
        sampled_values = np.zeros(arm_pool_size) 
        
        i = 0
        for arm in self.arms_pool['arm_id']:
            arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm][0] 
            
            S_i = self.arms_payoff_vectors["cumulated_rewards"][arm_pos] 
            F_i = self.arms_payoff_vectors["tries"][arm_pos] - S_i 
            
            sampled_values[i] = np.random.beta(S_i + 1, F_i + 1)
            
            i += 1

        rentability_score = sampled_values / self.arms_payoff_vectors["cost"]

        sorted_rentability_score_list = np.argsort(rentability_score)[::-1]

        budget_restant = self.budget

        profitability_threshold = 0

        for ids in sorted_rentability_score_list :
            arm_cost = self.arms_payoff_vectors["cost"][ids]
            

            if budget_restant - arm_cost >= 0 :
                budget_restant -= arm_cost
            else : 
                profitability_threshold = rentability_score[ids]
                break
                
        all_probabilities = np.zeros(len(self.arms_pool['arm_id']))

        for i, arm in enumerate(self.arms_pool['arm_id']):
            arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm][0]
            arm_rentability_score = rentability_score[arm_pos]
            arm_cost = self.arms_payoff_vectors["cost"][arm_pos]

            if arm_rentability_score > profitability_threshold :
                all_probabilities[i] = 1

            elif arm_rentability_score ==profitability_threshold :
                all_probabilities[i] = budget_restant / arm_cost
            else :
                all_probabilities[i] = 0

        chosen_arms = []

        for i, arm in enumerate(self.arms_pool['arm_id']):
            if np.random.uniform() < all_probabilities[i]:
                chosen_arms.append(arm)

        self.arm_chosen = chosen_arms

        return chosen_arms

        # -------------------------------------------------------------------

    def evaluate(self, observation):

        rewards = {}
        for arm_id in self.arm_chosen:
            feedback_rows = observation["feedback"][observation["arm_id"] == arm_id]
            if len(feedback_rows) > 0:
                feedback = feedback_rows.iloc[0]
                reward = 1 if feedback >= self.threshold else 0
                rewards[arm_id] = reward
        return rewards


        # -------------------------------------------------------------------

    def update(self, observation):

        rewards = self.evaluate(observation)
        
        for arm_id, reward in rewards.items():

            arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm_id][0]
            if reward > 0:
                self.arms_payoff_vectors["cumulated_rewards"][arm_pos] += reward
            self.arms_payoff_vectors["tries"][arm_pos] += 1
            self.price += self.arms_payoff_vectors["cost"][arm_pos]


                  
        
        # -------------------------------------------------------------------

    # =======================================================================