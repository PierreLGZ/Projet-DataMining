# Ranking-and-targeting
Project of Clément MALVY and Pierre LE GALEZE

## Description of the database
The database "data_avec_etiquettes.txt" contains 200 variables (V1...V200) and 494 021 observations.
V200 is the target variable, it has 23 modalities.
V1...V199 are the potential explanatory variables. Some are qualitative (V160, V161, V162), the others are quantitative.

## Objectives of the study
1. Produce a ranking system:
  a. That most accurately predicts the values of the target
  target variable V200.
  b. State the number of misclassifications if your model were applied
  on a deployment base with 4,898,424 obs.
2. Produce a "scoring" system:
  a. That targets the "m16" modality (the positives) of V200
  b. Announce the number of positives among the 10,000 observations that
  have the highest scores in a database of 4,898,424 obs.
3. Construct a modified target variable "V200_Prim" via the grouping of the modalities of "V200" by arguing your approach (number of groups, constitution of groups). Construct a model that predicts this new target variable "V200_Prim" as accurately as possible.

## Choice of methods and tools
Choice of methods to be used is free provided that :  
a. The final model can be expressed using a rule-based system applied to the initial variables of the dataset.  
b. (and/or) The model can be expressed using a linear combination based on
based on the initial variables (or coded 0/1 for qualitative explanatory variables).  

In all cases (objectives 1, 2 and 3), I expect a selection of variables to be made: only relevant variables are kept, and there are indications of their importance in the models to be deployed.  
Choice of tools (software) to be used is free as long as they are freely available and I am able to reproduce your calculations.

## Deliverables
A report outlining your approach for each objective (1, 2 and 3). A typical plan would be to adopt the presentation mode recommended by the CRISP-DM methodology, see in particular "The CRISP-DM outputs". For each objective, it is important to identify :  
a. which approaches you have tested, how you have identified the final model ;  
b. how the final model is expressed;  
c. which variables are included in the final model, with a ranking according to their
c. what variables are included in the final model, with a ranking of their importance;  
d. how did you estimate the performance based on the deployment?  
e. for objective (3), what is the strategy adopted to perform the
grouping of classes.  

A program (R, Python, or other) for each objective (1, 2, and 3) that can be applied to a deployment base (DB) in text format with tab separator of 4,898,424 obs. described exclusively by variables (V1...V199; variable names are in the header of each field). They must:
1. Take as input DB and produce a prediction stored in a text file named "predictions.txt" in the current directory.
2. Take as input DB and produce the score of membership to the class
"m16" in a text file named "scores.txt" in the current directory.
3. Take as input DB and a file " classes.txt " containing the original membership classes (variable name : V200). It should perform the grouping according to your strategy, make the prediction on DB, and save the predictions and the grouped classes in a file named "outputs.txt".

## Evaluation criteria
- Work to be done in groups of 3 students max.  
- Predictive performance.  
- Conformity of the predicted performances with the performances actually measured
measured during the correction.  
- Quality and reliability of the deployment programs (attention, size of the
deployment ≈ 2GB, make tests by duplicating the base at your disposal).  
- Argumentation of choices, positioning of different alternatives, relevance of variable selection.  
- Readability of predictive models, identification of relevant variables.  
- Quality of report writing (text, tables, graphs). Write
correctly (say 15 pages max to give an idea).  
