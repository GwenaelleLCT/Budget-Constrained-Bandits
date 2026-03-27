'''
Created on 24 mars 2026
'''

#--------------------------------------------------------------------#
#                                                                    #
#                          external imports                          #
#                                                                    #
#--------------------------------------------------------------------#

import numpy as np


#--------------------------------------------------------------------#
#                                                                    #
#                        Functions & Objects                         #
#                                                                    #
#--------------------------------------------------------------------#


class CTS():

    def __init__(self, arms=None, dimension_context=None): 
        
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "CTS"

        self.arms_payoff_vectors = {"cumulated_rewards" : np.zeros(len(self.ground_arms)),
                                    "tries" : np.zeros(len(self.ground_arms))
                                    }
        
        self.arm_chosen = None
        self.threshold = 4

        # Paramètre de variance
        self.v = 0.01
        # dimension du contexte 
        self.dimension_context = dimension_context
        # Initialisation des matrices A et vecteurs b pour chaque bras
        self.A = np.array([np.identity(self.dimension_context) for arm in range(len(self.ground_arms))]) 
        self.b = np.array([np.zeros(self.dimension_context) for arm in range(len(self.ground_arms))]) 
        
        
        # -------------------------------------------------------------------

    def run(self, observed_value, user_context=None):
        
        # Contexte
        self.current_context = user_context

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
            A_inv = np.linalg.inv(self.A[arm_pos])
            
            X = A_inv @ self.b[arm_pos]
            
            Y = (self.v ** 2) * A_inv
            
            sampled_theta = np.random.multivariate_normal(X, Y)
            
            x = self.current_context
            sampled_values[i] = sampled_theta @ x
            
            i += 1
            
        arm_chosen_index = np.argmax(sampled_values) 
        arm_chosen = self.arms_pool["arm_id"][arm_chosen_index]
            
        return arm_chosen

        # -------------------------------------------------------------------

    def evaluate(self, observation):

        reward = 0
        feedback = observation["feedback"][observation["arm_id"] == self.arm_chosen].iloc[0]
        if feedback >= self.threshold:
            reward = 1

        return reward

        # -------------------------------------------------------------------

    def update(self, observation):

        observed_reward = self.evaluate(observation)
        
        arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == self.arm_chosen][0]
        x = self.current_context
        
        # Mise à jour de A et b pour le bras choisi
        self.A[arm_pos] += np.outer(x, x)
        self.b[arm_pos] += observed_reward * x
        
        self.arms_payoff_vectors["cumulated_rewards"][arm_pos] += observed_reward
        self.arms_payoff_vectors["tries"][arm_pos] += 1
                  
        
        # -------------------------------------------------------------------

    # =======================================================================