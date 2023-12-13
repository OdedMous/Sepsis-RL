# Sepsis-RL

## Goal

In this project, I reimplemented the methodology from the paper [“The Artificial Intelligence Clinician Learns Optimal Treatment Strategies for Sepsis in Intensive Care”](https://www.nature.com/articles/s41591-018-0213-5) by Komorowski et al. 

- This work utilized an offline Reinforcement Learning algorithm to discover treatment strategies for septic patients in ICUs that may improve their chances of survival

- The methodology includes discretizing the state and action spaces into finite sets and then applying the Policy Iteration algorithm for learning an optimal value function. Model evaluation is conducted using the Weighted Importance Sampling method.


 ![Sepsis-RL](https://github.com/OdedMous/Sepsis-RL/blob/main/images/RL%20framework.png) 
*Image Source: https://www.jmir.org/2020/7/e18477/* 


## Results

- The full report can be found in: ```Project Report.pdf```

The figure below shows the estimated policy value of the clinicians’ actual treatments, the AI policy, a random policy, and a zero-drug policy, across 100 realizations of the environment based. It can be seen that in terms of expected value, the best AI policy performs much better than the clinicians' policy. In addition, for most of the realizations, the zero-drug policy outperforms the AI policy.

![Sepsis-RL](https://github.com/OdedMous/Sepsis-RL/blob/main/images/Result1.png)


