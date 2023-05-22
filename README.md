# Clean Growth
For Arkolakis and Walsh (2023). An algorithm to construct a power grid model from Open Street Map data with accompanying figures and analysis.

<p align="center">
  <img src="https://user-images.githubusercontent.com/74945619/100048710-77764d80-2de3-11eb-9c6b-8255d914309d.png" 
       alt="US grid map" 
       width="700"/>
</p>


# Introduction

This project is part of Arkolakis and Walsh (2023).

# Contents

The code is divided into cells. Some cells, particularly in the steps tuning the hyperparameters for logistic regression, are seperate because they have long run times. These computationally expensive cells are indicated with comments. 

We are not trying to establish causal inference with this kind of data, however we still want to make sure that we aren't ignoring any notable correlations. Thus we start with a large linear regression with as many controls as are feasible with the dataset and eliminate those that are not economically and/or statistically instructive. The logistic regression is for trying to understand what features help for predictions; it is not being used to describe the real world. Thus we can balance the data by oversampling. We try two feature selection methods to ensure the results are robust. 

