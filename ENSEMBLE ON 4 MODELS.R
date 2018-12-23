#ENSEMBLE ON 4 MODELS IN R,DECISION TREES, RANDOM FOREST, KNN AND GLM.

loan.data <- read.csv(file.choose(), header=T , sep=",")
library(caret)
library(e1071)
str(loan.data)
# Lets evaluate if Isolated model does better or a combo of models.

# Missing Values
sapply(loan.data, function(x)sum(is.na(x)))
# Doing Median Imputation of Value and we will scale the data too...
preprocess.loan <-  preProcess(loan.data,method= c("center" , "scale" ,
                                                   "medianImpute"))
# New data Frame
newloan <-  predict(preprocess.loan , loan.data)

sapply(newloan , function(x) sum(is.na(x)))
# Split the data into train and test
train_part <- createDataPartition(newloan$Loan_Status , p=0.75 , list=F)

loan.train <-  newloan[train_part,]
loan.test <-  newloan[-train_part,]

#Control the computational nuances of the train function
tcont <- trainControl(loan.train , method = "cv" , number = 5 ,
                      savePredictions= "final" ,classProbs = T )

#Fit Predictive Models over Different Tuning Parameters on our decision tree model

dt.model <- train(loan.train[c("ApplicantIncome","CoapplicantIncome",
                               "LoanAmount","Loan_Amount_Term", "Credit_History")], 
                  loan.train[, "Loan_Status"], method = "rpart", trControl = tcont, 
                  tuneLength = 3)

# Test it on test data and make predictions
loan.test$dt_pred <-  predict(dt.model,loan.test[,c("ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term", "Credit_History")])

confusionMatrix(loan.test$Loan_Status, loan.test$dt_pred)
#Kappa : 0.5227, Accuracy : 0.8235

#Random Forest model

rf.model <- train (loan.train[c("ApplicantIncome","CoapplicantIncome" ,"LoanAmount","Loan_Amount_Term", "Credit_History")],loan.train[,"Loan_Status"] , method="rf" , trControl = tcont , tuneLength = 3)

# Test it on test data and make predictions
loan.test$rf_pred <- predict(rf.model , loan.test[,c("ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term", "Credit_History")])

confusionMatrix(loan.test$Loan_Status, loan.test$rf_pred)
#Kappa : 0.4458 , Accuracy : 0.7843   

# Logistic Regression

glm.model <- train(loan.train[c("ApplicantIncome","CoapplicantIncome",
                                "LoanAmount","Loan_Amount_Term", "Credit_History")], 
                   loan.train[, "Loan_Status"], method = "glm", trControl = tcont, 
                   tuneLength = 3)

# Test it on test data and make predictions
loan.test$glm_pred <-  predict(glm.model, loan.test[,c("ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term", "Credit_History")])

confusionMatrix(loan.test$Loan_Status,loan.test$glm_pred)
#Kappa : 0.5227   , Accuracy : 0.8235 

#K-NN model

knn.model <- train(loan.train[c("ApplicantIncome","CoapplicantIncome",
                                "LoanAmount","Loan_Amount_Term", "Credit_History")], 
                   loan.train[, "Loan_Status"], method = "knn", trControl = tcont, 
                   tuneLength = 3)

# Test it on test data and make predictions
loan.test$knn_pred <-  predict(knn.model , loan.test[,c("ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term", "Credit_History")])

confusionMatrix(loan.test$Loan_Status, loan.test$knn_pred)
#Kappa : 0.5005 , Accuracy : 0.8105

# 04 Models - Decision Tree and GLM are good whereas 
#RF and KNN are not good enough.

# Combine these models - Compute the Probabilities and 
# Predict the average , majority voting and weighted average.

loan.test$dt_pred_prob <-  predict(object = dt.model ,loan.test[,c("ApplicantIncome",
                                                               "CoapplicantIncome","LoanAmount","Loan_Amount_Term",
                                                                "Credit_History")],type="prob")

loan.test$glm_pred_prob <-  predict(object=glm.model,loan.test[,c("ApplicantIncome",
                                                                "CoapplicantIncome","LoanAmount","Loan_Amount_Term",
                                                                "Credit_History")],type="prob")

loan.test$knn_pred_prob <-  predict(object=knn.model,loan.test[,c("ApplicantIncome",
                                                                  "CoapplicantIncome","LoanAmount","Loan_Amount_Term",
                                                                  "Credit_History")],type="prob")

loan.test$rf_pred_prob <-  predict(object=rf.model ,loan.test[,c("ApplicantIncome",
                                                                 "CoapplicantIncome","LoanAmount","Loan_Amount_Term",
                                                                 "Credit_History")],type= "prob")

#Averaging the Values
loan.test$pred_avg <-(loan.test$dt_pred_prob$Y +loan.test$glm_pred_prob$Y +
                     loan.test$knn_pred_prob$Y + loan.test$rf_pred_prob$Y)/4

# If the prob values are greater than 0.5, we would take it as Y else N
loan.test$pred_avg <-  factor(ifelse(loan.test$pred_avg>0.5,"Y","N"))

# Cross Table between the Target var Values and the Pred_Avg
confusionMatrix(loan.test$Loan_Status, loan.test$pred_avg)
#Kappa : 0.5227  , Accuracy : 0.8235

#Majority Voting
loan.test$pred_majority <-factor(ifelse(loan.test$dt_pred=="Y" & loan.test$glm_pred=="Y","Y",
                                ifelse(loan.test$rf_pred=="Y" & loan.test$knn_pred=="Y","Y","N")))

confusionMatrix(loan.test$Loan_Status , loan.test$pred_majority)
#Kappa : 0.5227 ,   Accuracy : 0.8235    

# Weighted Average
# Weight of Predictions are higher for more accurate models - DT and GLM
# Lets assign 0.50 to DT and GLM and 0.25 for RF and KNN.

loan.test$wt_avg <-(loan.test$dt_pred_prob$Y*.50)+(loan.test$glm_pred_prob$Y*.25)+
                  (loan.test$knn_pred_prob$Y*.25)+(loan.test$rf_pred_prob$Y*.25)

loan.test$wt_avg <- factor(ifelse(loan.test$wt_avg>.50,"Y" , "N"))

confusionMatrix(loan.test$Loan_Status , loan.test$wt_avg)
# Kappa : 0.5227, Accuracy : 0.8235       

#Conclusion is : Decision tree and Logistic Regression models give
#the best kappa and accuracy as these scores were not affected by 
#other methods of calculating accuracy to produce better models.
















































