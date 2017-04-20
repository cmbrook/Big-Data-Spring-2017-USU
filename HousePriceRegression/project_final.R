require(dplyr)
library(stats)
library(Metrics)
library(glmnet)
library(ggplot2)
library(caret)
library(xgboost)
library(randomForest)
library(car)
library(reshape2)
set.seed(13579)
HP.train <- read.csv('train.csv', sep = ',', header = TRUE)
HP.test <- read.csv('test.csv', sep = ',', header = TRUE)
HP.train$Id <- NULL
HP.test.Id <- HP.test$Id
HP.test$Id <- NULL
HP.test$SalePrice <- replicate(nrow(HP.test), 0)
ntrain <- nrow(HP.train)
ntest <- nrow(HP.test)
HP.all <- rbind(HP.train, HP.test)

###
# input: the dataset to check whether missing values existed
# output: print out all the columns with missing values. 
# the printed info includes: the column name, the number of missing values, is numeric category? (TRUE of FALSE)
###
check_missing_values <- function(dataSet)
{
  for (i in seq(1,ncol(dataSet))) {
    num_NA = sum(is.na(dataSet[, i]))
    is_numeric = is.numeric(dataSet[,i]) 
    if (num_NA > 0) {
      print (paste0('column ', names(dataSet)[i],' has #missing values: ', num_NA, ' is numeric: ',is_numeric ))
    }
  }
}
###
# input: the dataset to check whether categorical variables existed
# output: print out all the columns which are categorical variables.
# the printed info includes: the categorical column name, the number of factors
###
check_categorical_variables <- function(dataSet)
{
  print('************************************************')
  print('Listing all variables with categorical values:')
  count <- 0
  for (i in 1:ncol(dataSet)) {
    if (!is.numeric(dataSet[,i])) {
      count <- count + 1
      factor_names <- paste(levels(dataSet[,i]), sep=" ")
      print (paste0(names(dataSet)[i], ' is categorical variable with number of factors: '))
      print (factor_names)
    }
  }
  print (paste0('Total categorical variables:' , count))
  print('************************************************')
}
##
# input: the dataset.
# this function deals with missing values.
# Specifically, we replace missing values of numeric variables by the column's mean
# for categorical variables, we created a new distinguished value, called "Missing".
# output: return the dataset in which we replaced all missing values.
##
deal_missing_values <- function(dataSet, ntrain = 1460, type = 1)
{
  # type is the replace method for numeric values. 
  #if type = 1 --> using mean to replace.
  #if type = 2 --> using median to replace.
  dataSet[] <- lapply(dataSet, function(x){
    # check if variables have a factor:
    if(!is.factor(x)) {
      #replace NA by mean
      x[is.na(x)] <- mean(x[1:1460], na.rm = TRUE) 
      # if (type == 1) x[is.na(x)] <- mean(x[1:1460], na.rm = TRUE) 
      # else if (type == 2) if (type == 1) x[is.na(x)] <- median(x[1:1460], na.rm = TRUE) 
    }
    else {
      # otherwise include NAs into factor levels and change factor levels:
      x <- factor(x, exclude=NULL)
      levels(x)[is.na(levels(x))] <- "Missing"
    }
    return(x)
  })
  
  return(dataSet)
}

##
# input: the dataset
# encoding categorical variables. If a variable has 3 levels --> we need 2 bit to encode
# output: return the dataset in which we encoded all the categorical variables
##
encode_categorical_variables <- function(dataSet, ntrain=1460)
{
  dataSetTemp <- dataSet
  for (i in 1:ncol(dataSet)){
    if (is.factor(dataSet[,i])) {
      col_name = names(dataSet)[i]
      t <- model.matrix(~ factor(dataSet[,i]) - 1)
      dataSetTemp <- cbind(t[,2:ncol(t)], dataSetTemp)
      new_col_names <- paste(col_name, "_bit", seq(1, ncol(t) - 1))
      names(dataSetTemp)[1:ncol(t) - 1] <- new_col_names
    }
  }
  removed <- c()
  for (i in 1:ncol(dataSetTemp)) {
    if (is.factor(dataSetTemp[,i])) removed <- c(removed, i)
  }
  dataSetTemp <- dataSetTemp[, -removed]
  return(dataSetTemp)
}

##
# input: the dataSet (HP.all) 
# this function will select the important variables in the training set
# we used the HP.all because after getting important variables in the training set,
# we can remove unimportant variables in the test set also.
# We used elastic net to select the important variables, we set alpha = 0.3 as it maximized 
# the prediction of the training set.
# output: the dataset in which we kept only important features.
##
do_variable_selection <- function(dataSet, alpha = 0, printImportantVars = FALSE)
{
  trainData <- dataSet[1:ntrain,]
  trainData$SalePrice <- log(trainData$SalePrice)
  trainData.cv.lasso = cv.glmnet(x = as.matrix(trainData[,1:(ncol(trainData) - 1)]),
                                 y=as.matrix(trainData[,ncol(trainData)]), alpha = 0.3)
  # trainData.cv.lasso = glmnet(x = as.matrix(trainData[,1:(ncol(trainData) - 1)]), 
  #                             y=as.matrix(trainData[,ncol(trainData)]), 
  #                             alpha = 1, lambda = 0.001)
  
  # co <- coef(trainData.cv.lasso, s = "lambda.1se")
  co <- coef(trainData.cv.lasso, s = "lambda.min")
  co <- co[2:nrow(co),] #remove the coefficient of intercept, which is the first row
  trainData <- trainData[, co > 0]
  vars <- names(trainData)
  if (printImportantVars) {
    print('Important variables:')
    print(vars)
  }
  if (!'SalePrice' %in% names(trainData)){
    vars <- c(vars, 'SalePrice')
  }
  dataSet <- subset(dataSet, select = vars)
  return(dataSet)
}

##
# input: dataSet and a boolean variable: do_transform_all (default = FALSE)
# if do_transform_all is set to TRUE, we will do log transform for all the predictors + response
# otherwise, we do log transformation on the response only.
# output: the dataset in which we performed log transformation
##
do_transform <- function(dataSet, do_transform_all = FALSE)
{
  if (do_transform_all) {
    for (i in 1:(ncol(dataSet) - 1) ) {
      if (min(dataSet[,i]) == 0 && max(dataSet[,i]) == 1 ) {
        #categorical values --> do nothing
      } else{
        dataSet[,i] <- log(dataSet[,i])
      }
    }
  }
  dataSet[,ncol(dataSet)] <- log(dataSet[,ncol(dataSet)])
  return(dataSet)
}
##
# input: the response
# Since we do log tranformation on the reponse,
# for the test set, we need to do reversed transformation for the response.
# output: the reponse in which revesed transformation is performed
##
reverse_transformed_response <- function(response)
{
  return(exp(response))
}

build_model <- function(train, type = 2, transformed = FALSE)
{
  
  if (type == 1) {
    #multiple linear regression model
    fit <- lm(SalePrice ~., train)  
    fit$rsquared <- summary(fit)$r.squared
    fit$adj.rsquared <- summary(fit)$adj.r.squared
    fit$predicted <- fit$fitted.values
    
  }
  if (type == 2) {
    #add elastic net regularization to reduce model overfitting
    # fit = cv.glmnet(x = as.matrix(train[,1:(ncol(train) - 1)]), y=as.matrix(train[,ncol(train)]), type.measure = "mse")
    # plot(fit)
    set.seed(1024)
    fit = glmnet(x = as.matrix(train[,1:(ncol(train) - 1)]), y=as.matrix(train[,ncol(train)]), family="gaussian", alpha = 0.1, lambda = 0.1)
    # plot(fit, xvar = "dev", label = TRUE)
    # plot(fit)
    # print(fit$lambda.min)
    # print(fit$lambda.lse)
    fit$predicted <- predict(fit, newx = as.matrix(train[,1:(ncol(train) - 1)]),  type="response", family="gaussian", standardize=TRUE)
  }
  if (type == 3) {
    # random forest in here:
    fit <- randomForest(x = as.matrix(train[,1:(ncol(train) - 1)]), y = as.matrix(train[,ncol(train)]), 
                        ntree = 200, mtry = 10)
    plot(fit)
    # fit$predicted <- as.vector(fit$predicted)
  }
  if (type == 4) {
    dataAll <- xgb.DMatrix(data=data.matrix(train[,1:(ncol(train)-1)]),
                           label=data.matrix(train[,ncol(train)]))
    # #xgboost here
    # param <- list(max.depth=12,eta=0.01,nthread = 8, silent=1,objective='reg:linear', alpha=1)
    # #param <- list(max.depth = 4,verbose = FALSE, eta = 0.1, nthread = 8, objective = "binary:logistic",task="pred")
    
    # 
    # res <- xgb.cv(params = param, data = dataAll,  nrounds = 1000, nfold = 10, 
    #               prediction = TRUE, eval_metric="rmse", verbose = FALSE)
    # max_round = which.min(res$dt[, test.rmse.mean])
    # print(max_round)
    # rmse <- res$dt[, test.rmse.mean][max_round]
    # 
    # fit <- xgboost(params = param, data=dataAll, nrounds = max_round, verbose=FALSE)
    
    xgb_params = list(
      booster = 'gbtree',
      objective = 'reg:linear',
      colsample_bytree=1,
      eta=0.005,
      max_depth=12,
      min_child_weight=3,
      alpha=0.3,
      lambda=0.4,
      gamma=0.01,
      subsample=0.6,
      seed=5,
      silent=TRUE)
    fit = xgb.train(xgb_params,dataAll, nrounds = 2000)
  }
  if (type != 4) {
    fit$actual <- train$SalePrice
    if (transformed) {
      #reconvert
      fit$predicted <- reverse_transformed_response(fit$predicted)
      fit$actual <- reverse_transformed_response(fit$actual)
    }  
    
    fit$rmse <- rmse(actual = fit$actual, predicted  = fit$predicted)
    fit$mse <- mse(actual = fit$actual, predicted  = fit$predicted)
    print(paste0('RMSE of training :',fit$rmse))
  }
  return(fit)
}


#missing values: categorical + numeric values. Numeric values can be replaced by mean, median for ease.
#outliers: do later.
#do regression: variable selection + linear regression + xgboost + random forest
# check the residuals
# Mine: regression part + transformation 

####################### 1. Dealing with missing values for training and testing data #################################################
check_missing_values(HP.all)
HP.all <- deal_missing_values(HP.all, ntrain,type = 2)
print ('check missing values of the dataframe after replacing missing values. Should print blank')
check_missing_values(HP.all)

####################### 2. Encoding categorical variables ############################################################################
check_categorical_variables(HP.train)
# HP.train <- encode_categorical_variables(HP.train)
HP.all <- encode_categorical_variables(HP.all)
print ('check categorical values of the dataframe after encoding. Should print blank')
check_categorical_variables(HP.all)

#reconstruct some variables:
HP.all <- mutate(HP.all,
                 house_age = YrSold + MoSold/12 - YearBuilt,
                 remodel_house_age = abs(YearRemodAdd - YearBuilt),
                 gara_age = abs(GarageYrBlt - YearBuilt),
                 sold_year_to_now = 2017 - YrSold - MoSold/12,
                 built_year_to_now = 2017 - YearBuilt
                 
)
SalePrice <- HP.all$SalePrice
HP.all$SalePrice <- NULL
HP.all$SalePrice <- SalePrice
HP.all$GarageYrBlt <- NULL
HP.all$YearBuilt <- NULL
HP.all$YearRemodAdd <- NULL
HP.all$YrSold <- NULL
HP.all$MoSold <- NULL
#************************************************************************************************************************************#
best_alpha = 0.0
min_rmse = 1
for (alpha in seq(0,1,0.1)) {
  HP.all.selected <- do_variable_selection(HP.all, alpha = alpha)
  HP.all.selected <- do_transform(HP.all.selected)
  
  HP.train = HP.all.selected[1:ntrain,]
  HP.train <- HP.train[-c(463,524,633,826,1299,1325) , ]
  HP.test = HP.all.selected[(ntrain+1) : nrow(HP.all.selected),]
  HP.test$SalePrice <- NULL
  
  # fit one:
  model <- lm(SalePrice~., HP.train)
  fitone.lm.rmse = rmse(actual = HP.train$SalePrice, predicted = predict(model, HP.train))
  print(paste0('alpha = ', alpha, ' -> fit one -> ', fitone.lm.rmse))
  #build 10 fold cross validation for HP.train
  HP.train.yval = rep(0, nrow(HP.train))
  xvs=rep(1:10, length=nrow(HP.train))
  xvs=sample(xvs)
  lm.models <- c()
  for (i in 1:10){
    train = HP.train[xvs != i,]
    test = HP.train[xvs == i,]
    lm.model <- lm(SalePrice ~., train)
    HP.train.yval[xvs == i] = predict(lm.model, test)
    lm.models <- c(lm.models, lm.model)
  }
  lm.RMSE <- rmse(actual = HP.train$SalePrice, predicted = HP.train.yval)
  print(paste0('alpha = ', alpha, ' -> 10 folds -> ', lm.RMSE))
  if (lm.RMSE < min_rmse) {
    min_rmse = lm.RMSE
    best_alpha = alpha
  }
}
print(paste0('best_alpha = ', best_alpha))
HP.all.selected <- do_variable_selection(HP.all, alpha = best_alpha)
HP.all.selected <- do_transform(HP.all.selected)

HP.train = HP.all.selected[1:ntrain,]
HP.train <- HP.train[-c(463,524,633,826,1299,1325), ]
HP.test = HP.all.selected[(ntrain+1) : nrow(HP.all.selected),]
HP.test$SalePrice <- NULL
#build 10 fold cross validation for HP.train
# HP.train.yval = rep(0, nrow(HP.train))
# xvs=rep(1:10, length=nrow(HP.train))
# xvs=sample(xvs)
# PredictedSalePrice <- rep(0, nrow(HP.test))
# for (i in 1:10){
#   train = HP.train[xvs != i,]
#   test = HP.train[xvs == i,]
#   lm.model <- lm(SalePrice ~., train)
#   HP.train.yval[xvs == i] = predict(lm.model, test)
#   PredictedSalePrice = PredictedSalePrice + predict(lm.model, HP.test)
# }
# #predict for the test:
# PredictedSalePrice = PredictedSalePrice/10
# HP.test$SalePrice = reverse_transformed_response(PredictedSalePrice)

#fit one model
lm.model <- lm(SalePrice~., HP.train)
HP.test$SalePrice <- reverse_transformed_response(predict(lm.model, HP.test))
fitone.lm.rmse = rmse(actual = HP.train$SalePrice, predicted = predict(lm.model, HP.train))
print(fitone.lm.rmse)
#write to file
submission <- data.frame(Id <- HP.test.Id, SalePrice <- HP.test$SalePrice)
names(submission) <- c('Id', 'SalePrice')
write.csv(file = 'submission_lm.csv', x = submission, row.names = FALSE)

# print ('Building glmnet model')
# HP.train.glmnet_model <- build_model(HP.train, type = 2)
# 
# 
 
# print ('Building random forest model')
# lm.model <- lm(SalePrice~., HP.train)
# HP.test$SalePrice <- predict(lm.model, HP.test)
# nrow_test <- nrow(HP.test)
# nrow_train <- nrow(HP.train)
# HP.combined <- rbind(HP.train, HP.test)
# HP.combined.rf_model <- build_model(HP.combined, type = 3)
# HP.test$SalePrice <- reverse_transformed_response(predict(HP.combined.rf_model, HP.test))
# submission <- data.frame(Id <- HP.test.Id, SalePrice <- HP.test$SalePrice)
# names(submission) <- c('Id', 'SalePrice')
# write.csv(file = 'submission_rf.csv', x = submission, row.names = FALSE)

print ('Building xgboost model')
lm.model <- lm(SalePrice~., HP.train)
HP.test$SalePrice <- predict(lm.model, HP.test)
nrow_test <- nrow(HP.test)
nrow_train <- nrow(HP.train)
HP.combined <- rbind(HP.train, HP.test)

HP.combined.xgb_model <- build_model(HP.combined, type = 4)
fitone.xgb.rmse = rmse(actual = HP.combined$SalePrice, predicted = predict(HP.combined.xgb_model,
                                                                        xgb.DMatrix(data=data.matrix(
                                                                          HP.combined[,1:(ncol(HP.combined)-1)]))))
print(fitone.xgb.rmse)
HP.test$SalePrice <- reverse_transformed_response(
                          predict(HP.combined.xgb_model, xgb.DMatrix(data=data.matrix(
                            HP.test[,1:(ncol(HP.test)-1)]))))
submission <- data.frame(Id <- HP.test.Id, SalePrice <- HP.test$SalePrice)
names(submission) <- c('Id', 'SalePrice')
write.csv(file = 'submission_xgb.csv', x = submission, row.names = FALSE)
