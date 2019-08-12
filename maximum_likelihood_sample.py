# -*- coding: utf-8 -*-

"""
likelihood
target label is defined as [0, 0, 0, 1]
probability of male(1) is w. the other one of female(0) is (1 - w)
the most highest likelihood is calculated.
"""

import numpy as np
import matplotlib.pyplot as plt

# sample data
male_prob = 0.0
male_prob_list = []
while male_prob <= 1.0:
    male_prob_list.append(male_prob)
    male_prob += 0.05

# likelihood
likelihood_list = []
max_likelihood = 0.0
max_index      = 0
for i, prob in enumerate(male_prob_list):
    likelihood = ((1 - prob)**3) * prob
    likelihood_list.append(likelihood)
    if max_likelihood < likelihood:
        max_likelihood = likelihood
        max_index      = i
max_male_prob = male_prob_list[max_index]

# plot graph
plt.figure(figsize=(6, 6))
plt.plot(male_prob_list, likelihood_list, c='blue')
plt.title('Max likelihood:'+str(max_likelihood)+', Male Probability:'+str(max_male_prob))
plt.grid(True)
plt.xlim([0.0, 1.0])
plt.xlabel('Male probability')
plt.ylabel('Likelihood')
plt.show()