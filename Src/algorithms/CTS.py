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

    def __init__(self, arms=None, context_example=None): 
        
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "CTS"

        nb_arms = len(self.ground_arms)
        nb_dimensions = context_example.shape[1] - 1
        self.arms_payoff_vectors = {"cumulated_rewards" : np.zeros(nb_arms),
                                    "tries" : np.zeros(nb_arms),
                                    "covariance_matrix" : np.zeros((nb_arms, nb_dimensions, nb_dimensions)),
                                    "inverse_covariance_matrix" : np.zeros((nb_arms, nb_dimensions, nb_dimensions)),
                                    "features_matrix" : np.zeros((nb_arms, nb_dimensions,)),
                                    "theta_hat" : np.zeros((nb_arms, nb_dimensions,))
                                    }
        
        self.arms_payoff_vectors["covariance_matrix"][:] = np.eye(nb_dimensions).astype(float)
        self.arms_payoff_vectors["inverse_covariance_matrix"][:] = np.eye(nb_dimensions).astype(float)
            
        
        self.arm_chosen = None
        # Paramètre de variance
        self.v = 0.5

        self.threshold = 4

        
        # -------------------------------------------------------------------

    def run(self, observed_value, user_context=None):
        
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
        expected_payoff = np.zeros(arm_pool_size)

        i = 0
        for arm in self.arms_pool['arm_id']:
            arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm][0]
             
            self.arms_payoff_vectors["theta_hat"][arm_pos] = np.random.multivariate_normal( np.dot(self.arms_payoff_vectors["inverse_covariance_matrix"][arm_pos], self.arms_payoff_vectors["features_matrix"][arm_pos]), \
                                                       (self.v ** 2) * self.arms_payoff_vectors["inverse_covariance_matrix"][arm_pos])
            
            expected_payoff[i] = np.dot(self.arms_payoff_vectors["theta_hat"][arm_pos].T, self.current_context)
            
            i += 1
            
        arm_chosen_index = np.argmax(expected_payoff) 
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
        self.arms_payoff_vectors["cumulated_rewards"][self.arm_chosen] += observed_reward
        self.arms_payoff_vectors["tries"][self.arm_chosen] += 1
        
        self.arms_payoff_vectors["covariance_matrix"][self.arm_chosen] += np.outer(self.current_context, self.current_context)
        self.arms_payoff_vectors["inverse_covariance_matrix"][self.arm_chosen] = np.linalg.inv(self.arms_payoff_vectors["covariance_matrix"][self.arm_chosen])
        self.arms_payoff_vectors["features_matrix"][self.arm_chosen] += observed_reward * self.current_context
                  
        
        # -------------------------------------------------------------------

    # =======================================================================