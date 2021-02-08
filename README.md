# Predicting-patients-readmission-in-the-hospital

There are two files where the the first file is of prediction where I did not use diagnoses codes and other the file is with Diagnoses codes.

I have used the Real time hospital data of patients which is really messy with 26 tables. It has been accessed by completing a certificaion under CITI group. The dataset is 
named as MIMIC III data.




Readmission of a patient in the Hospital within next 30 days

Abstract
The database used in this is MIMIC III (Medical Information Mart for Intensive Care), which is freely available, comprised of health-related data of around 40000 patients stayed under critical care from 2001 to 2012. “The database has information like demographics, vital sign measurements made at the bedside (~1 data point per hour), laboratory test results, procedures, medications, caregiver notes, imaging reports, and mortality (both in and out of hospital).” [1]
Only 3 tables from the MIMIC III database has been used such as ADMISSIONS, NOTEEVENTS and DIAGNOSES_ICD. Preprocessing the tables included creation of a response variable for the problem statement from the existing predictors. For that I used ADMISSIONS table. Then from DIAGNOSES_ICD table taken the ICD codes and assigned them to their respective categories and merged with the ADMISSIONS table. Then from NOTEEVENTS preprocessed the text data of each patient by counting the number of words from each patient and merged them with to create the final dataset for my modelling.
To check my model’s accuracy ran the base models with default parameters. At this point of time used Logistic Regression, Random Forest and Neural Network. Then tuned the hyperparameters for both the models to compare the prediction results from the previous results. Even checked the calibration curves of both the models which were not satisfactory. So calibrated the models and plotted the calibration curves. Turned out Random forest is the best predictor of both.



Introduction:

Healthcare is the main concern in the world nowadays. And there is a lot of data in this area like patient records, test samples, test reports and patient information. It is estimated that approximately 30 percent of the world’s warehouse data comes from the industry of healthcare. It is important to model these data for better experience and to provide proper treatment for the patient from the healthcare department. 
To obtain this, modeling the data is required based on the patient records and estimating their possibility of readmitting. There are many such projects, built using data science in healthcare. For e.g.: Medical Image analysis for better treatment of patients based on previous data, Creation of drugs and Virtual assistance for patients and customer support etc.

Objective:
The objective of this project is to determine the patients’ those should readmit in the hospital within next 30 days. As It’s very important for a hospital and a doctor to know whether a patient will be readmitted in any of the hospitals. Based on that they can change the medication, in order to prevent them from being readmitted.

The Dataset:
The dataset has been created by merging 3 tables from the MIMIC III database. The final dataset that was created has 34560 observations and 3019 columns. The main columns are Subject_ID, Readmission, ICD codes and the words that were extracted using NLP from Noteevents table.

 
The above snippet shows that all the columns were numerical and also first 3 observations from the dataset.
Data Preprocessing:
ADMISSIONS: The main columns of this table are Subject_ID is the patient ids of the each patient, HADM_ID is the id of patient for each admission in the hospital, ADMITTIME is the patient’s admission date and time, DISCHTIME is the patient’s discharge date and time, DEATHTIME is the patient’s death date and time and ADMISSION_TYPE is the type of a patients admission while admitting in the hospital such as EMERGENCY, URGENT, ELECTIVE and NEWBORN.
 As the prediction is the readmission of a patient within 30 days, only emergency cases are considered by removing Newborn for now. Urgent and Emergency comes under one category. 
 

Then the patients who were dead are also removed from the table. 
 
Then converted the date and time columns to the proper date and time format for better processing for the data further.
 
Then sorted every SUBJECT_ID with their respective admission times from ADMITTIME column. Now to work towards the response variable adding the next admission date and admission type for each subject using group by as the dates of all the patients are different. I am shifting the data by 1 here. Coming to the ELECTIVE type of admission, filling Nan and Nat in the admission type and time columns which were newly added to the dataframe as READMITTIME and READMISSION_TYPE. Then backfilling with respect to each patient if they have emergency to be filled in place of null values created.
Added a new column DAYS_NEXT_ADMIT which is a difference of READMITTIME and DISCHTIME. Then again converted into number of days.

DIAGNOSES_ICD: The main columns in this table are SUBJECT_ID, HADM_ID, and the ICD9_CODEs which describes the type of disease or symptoms. 
 
The ICD9 codes here are more than 3 digits which describes the exact symptoms. But we only need 3 digits to code the symptoms which will be used for our prediction further. So created a new column to store these.
 
Then recoded the numbers based on the category of ICD9 code and clubbed every admissions multiple observations into one.
 
Now created dummies for theses symptoms and joined it with the previous preprocessed column.
 

NOTEEVENTS: The main columns in this table is SUBJECT_ID, HADM_ID, CHARTDATE is the date when the notes were given to a patient during their admission, CHARTTIME is the time when the notes were given to a patient during their admission, TEXT is the column that has the notes, any medical procedures or any medication suggested or given to the admitted patient. There are various categories of these notes: 
  
A single admitted patient’s note looks like:
 
Removed all the dead patients from this table as well like I did earlier from admissions table. Then concatenated all the notes of a single patient for their multiple admissions in the hospital after sorting those by date and time. And also kept the first entry of each patient by removing any other duplicates. After this merged this data set with the previous dataset with all required columns.
Now created the response variable for those who those who readmits before 30 days as 1 and rest as 0 using DAYS_NEXT_ADMIT column. Below is the distribution of the output label
 
So, with the text data it is difficult to feed these into the algorithm for prediction. So, used Natural Language Processing to create a matrix of the count of each word’s occurrence for each patient from the notes. And created the final dataframe for the predictons.
 

The output label frequency distribution:
 
 There are not enough readmitted cases where the data is clearly imbalanced. To deal with this while modelling I have set the parameter Class_weight as balanced. Now dividing the dataset into X_train and X_test while assigning 20% of dataset to test. Even separating the response variable such as y_train and y_test.


Modelling:
Before starting with this lets discuss about confusion matrix and which error values can be accepted if increased but the other type should be decreased. Lets say X is applying for a H-1B Visa:
True positive (TP): Prediction is X should Readmit and X did, we want that
True negative (TN): Prediction is X should not Readmit and X did not, we want that too
False positive (FP): Prediction is X should Readmit and X did not, false alarm, bad
False negative (FN): Prediction is X should not Readmit and X did, the worst
Now We know that in case of FNs, the person gets false hope of getting a Visa while in reality he doesn’t and in case of FPs person is said that he won’t get Visa but eventually he gets that acceptable for some extent. Now, lets try to look at the math which can give us less FNs but at the cost of more FPs.
Recall/Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP)
So, after looking at the math we can decide to have more Recall and less specificity in general. Both should be balanced as we want FN to decrease with the cost of increase in FPs. I am tuning the models for better Recall using RandomizedSearchCV from SKLearn.
 
Logistic regression, a statistical model that uses a logistic function to model a binary dependent variable. It predicts the probabilities of both the classes in this case. Default threshold is set to 0.5 to differentiate the classes among each other.
First of all, the basic Logistic regression is used without any tuned parameters to see model’s prediction power. 
 
Upon running this model, the prediction values are as follows:
 
  
The above values are quite impressive for now as we see that AUC is 84.4% which says that we can adjust the threshold to decrease the False negatives. In our case we need less number of False negatives and it vary from problem to problem.

Now after tuning the class weight and cost parameters. The best parameters were balanced class weight and 0.1 cost. And while tuning I used scoring as Recall in RandomizedSearchCV which would give the best Recall of all the parameters passed.
 
After using this new model for prediction, we get:
 
 The Recall dropped and the AUC values than before. So, here we will go with the previous models’ predictions as best as of now.

Now Coming to Random Forest, it’s a supervised machine learning algorithm which randomly creates and combines multiple decision trees as one forest. This algorithm does not rely on a single learning model rather tries to improve accuracy using different trees from decision trees. The root nodes feature splitting is done randomly in this algorithm which is why its name states Random forest.
Just like previous model, applied Random Forest without tuning to get the prediction metrics.
 

Results of the predictions: 
  
Though the accuracy and AUC values are promising let’s try to improve the models Recall which is our main aim. So in the next step will tune the model.
Now tuning the model parameters with the below given values:
 
After tuning the best parameters are:
 
So applied the above parameters in the new model to check the new prediction metrics.
  
While comparing the results with all the models we could clearly say that tuned Random Forest is the best predictor of all the models. This gives us high Recall by reducing the FNs as we decided this is achieved at the cost of FPs. And AUC value is also good. 
Top Predictors after the tuning the Random Forest model:
 

Deep neural Network: In this project I have tried to implement a deep neural network with 2 dense layers and 1 output layer. The 1st dense layer has 180 neurons which receives the input from all the predictors and has been deployed with the RELU activation function. The 2nd layer 110 neurons which receives the input from the output data of all 180 neurons connected to each neuron in 2nd layer. Even this layer has RELU as the activation function. The last layer also gets input from the output values of 2nd layer thus has 2 layers as our output labels are of 2 levels. 
While compiling the model ADAM and Sparse Categorical Crossentropy are used as optimizer and loss function respectively. Upon training the model the prediction values are as follows:
 
 The results show low Recall but has good accuracy and AUC value.

Model Calibration:
In model calibration we try to improve our model such that the distribution and behavior of the probability predicted is like the distribution and behavior of probability observed in training data.
 
This is the calibration curve for Logistic Regression and Random Forest before calibration.

After calibration using Sigmoid and Isotonic in Random forest:
 
 
This is the frequency distribution plot of all the probabilities predicted after calibration.
After calibration using Sigmoid and Isotonic in Logistic Regression:
 

 
This is the frequency distribution plot of all the probabilities predicted after calibration.

More Analysis:
Created a plot which shows the AUC performance of Random forest with 100,200,300……2000 trees on training and testing data.
 

The highest AUC was obtained somewhere around 1300 tress.
Now created a plot showing the AUC performance of Random forest with 100,200,300….3018 predictors on training and testing data.
 
The highest AUC was obtained somewhere around 1500 predictors.

To see how the size of the data and predictors both contributes to the AUC performance when provided to the model. To achieve this created a plot with learning curve where X-axis has number of predictors, Y-axis has size of training data and Z- axis has the AUC values. 
The predictors are passed 100, 200, 300….3018 and for each iteration of prediction all the different size of the data was iterated in the inner loop as 2700, 5400….27648. So, the loop runs for 310 times while capturing 310 different AUC values.
 

Model Comparison:
 
The tuned Random Forest outperforms all the models that were tried to predict the readmission of a patient in the hospital within next 30 days. As it has high Recall, Less False negatives, High Accuracy and AUC values as well.
