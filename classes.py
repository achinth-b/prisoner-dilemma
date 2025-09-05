import random
import numpy as np
from typing import Tuple, Dict, List

class Prisoner: 
    def __init__(self, p: float = 0.5): 
        # probability of defecting is defined as p 
        # the default is set to 0.5
        self.p = p
    
    def action(self) -> str: 
        # generate from a random uniform distribution with the seeded p as the policy 
        # for whether or not the prisoner decides to defect or cooperate
        rng = random.random()
        return 'defect' if rng <= self.p else 'cooperate'

    def set_probability(self, p: float):
        """Update the probability of defecting"""
        self.p = p

class Dilemma: 
    def __init__(self, payoff_matrix: Dict[Tuple[str, str], Tuple[int, int]] = None): 
        # Classic prisoner's dilemma payoff matrix 
        # (prisoner1_payoff, prisoner2_payoff)
        if payoff_matrix is None: 
            self.payoff_matrix = {
                ('cooperate', 'cooperate'): (3, 3),  # Both cooperate - mutual benefit
                ('cooperate', 'defect'): (0, 5),     # P1 cooperates, P2 defects - P1 gets sucker's payoff
                ('defect', 'cooperate'): (5, 0),     # P1 defects, P2 cooperates - P1 gets temptation payoff
                ('defect', 'defect'): (1, 1)         # Both defect - mutual punishment
            }
        else: 
            self.payoff_matrix = payoff_matrix
    
    def play_round(self, prisoner1: Prisoner, prisoner2: Prisoner) -> Tuple[int, int]:
        """Play one round of the prisoner's dilemma"""
        action1 = prisoner1.action()
        action2 = prisoner2.action()
        return self.payoff_matrix[(action1, action2)]
    
    def expected_payoff(self, p1: float, p2: float) -> Tuple[float, float]:
        """Calculate expected payoffs given probabilities of defecting"""
        # Calculate expected payoffs based on probability distributions
        # P(cooperate, cooperate) = (1-p1) * (1-p2)
        # P(cooperate, defect) = (1-p1) * p2
        # P(defect, cooperate) = p1 * (1-p2)
        # P(defect, defect) = p1 * p2
        
        prob_cooperate_cooperate = (1 - p1) * (1 - p2)
        prob_cooperate_defect = (1 - p1) * p2
        prob_defect_cooperate = p1 * (1 - p2)
        prob_defect_defect = p1 * p2
        
        expected_p1 = (prob_cooperate_cooperate * self.payoff_matrix[('cooperate', 'cooperate')][0] +
                      prob_cooperate_defect * self.payoff_matrix[('cooperate', 'defect')][0] +
                      prob_defect_cooperate * self.payoff_matrix[('defect', 'cooperate')][0] +
                      prob_defect_defect * self.payoff_matrix[('defect', 'defect')][0])
        
        expected_p2 = (prob_cooperate_cooperate * self.payoff_matrix[('cooperate', 'cooperate')][1] +
                      prob_cooperate_defect * self.payoff_matrix[('cooperate', 'defect')][1] +
                      prob_defect_cooperate * self.payoff_matrix[('defect', 'cooperate')][1] +
                      prob_defect_defect * self.payoff_matrix[('defect', 'defect')][1])
        
        return expected_p1, expected_p2