# Sepsis-RL

## Goal

Sepsis is a life-threatening medical condition that can lead to organ failure and death without timely treatment in the Intensive Care Unit (ICU). Managing septic patients involves the administration of several medications, but there is no universal treatment policy due to patient variability and the complexity of the disease.

In this project, I developed a Reinforcement Learning (RL)-based agent that adjusts medication doses based on real-time clinical information from patients. This is a reimplementation of the methodology presented in the paper [“The Artificial Intelligence Clinician Learns Optimal Treatment Strategies for Sepsis in Intensive Care”](https://www.nature.com/articles/s41591-018-0213-5) by Komorowski et al. 

The methodology includes discretizing the state and action spaces into finite sets and then applying the Policy Iteration algorithm for learning an optimal value function. Model evaluation is conducted using the Weighted Importance Sampling method.


 ![Sepsis-RL](https://github.com/OdedMous/Sepsis-RL/blob/main/images/RL%20framework.png) 
*Image Source: https://www.jmir.org/2020/7/e18477/* 


## Results

- The full report can be found in: ```Project Report.pdf```

The figure below shows the estimated policy value of the clinicians’ actual treatments, the AI policy, a random policy, and a zero-drug policy, across 100 realizations of the environment. It can be seen that in terms of expected value, the best AI policy performs much better than the clinicians' policy. In addition, for most of the realizations, the zero-drug policy outperforms the AI policy.

![Sepsis-RL](https://github.com/OdedMous/Sepsis-RL/blob/main/images/Result1.png)


