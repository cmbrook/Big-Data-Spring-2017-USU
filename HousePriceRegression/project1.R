require(dplyr)
library(stats)
library(Metrics)
library(glmnet)
library(ggplot2)
library(caret)
library(xgboost)
library(randomForest)
HP.train <- read.csv('train.csv', sep = ',', header = TRUE)
HP.test <- read.csv('test.csv', sep = ',', header = TRUE)
HP.train$Id <- NULL
HP.test.Id <- HP.test$Id
HP.test$Id <- NULL
HP.test$SalePrice <- replicate(nrow(HP.test), 0)
ntrain <- nrow(HP.train)
ntest <- nrow(HP.test)
HP.all <- rbind(HP.train, HP.test)
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

deal_missing_values <- function(dataSet, ntrain = 1460, type = 1)
{
  # type is the replace method for numeric values. 
  #if type = 1 --> using mean to replace.
  #if type = 2 --> using median to replace.
  dataSet[] <- lapply(dataSet, function(x){
    # check if variables have a factor:
    if(!is.factor(x)) {
      #replace NA by mean
      if (type == 1) x[is.na(x)] <- mean(x[1:1460], na.rm = TRUE) 
      else if (type == 2) if (type == 1) x[is.na(x)] <- median(x[1:1460], na.rm = TRUE) 
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

#
# encoding categorical variables. If a variable has 3 levels --> we need 2 bit to encode
#
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

deal_outliers <- function(dataSet)
{
  
}

do_variable_selection <- function(dataSet)
{
  #must return a dataframe in which important variables are kept.
    
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
                        ntree = 200, mtry = 100)
    plot(fit)
    # fit$predicted <- as.vector(fit$predicted)
  }
  if (type == 4) {
    #xgboost here
    param <- list(max.depth=2,eta=0.1,nthread = 8, silent=1,objective='reg:linear')
    #param <- list(max.depth = 4,verbose = FALSE, eta = 0.1, nthread = 8, objective = "binary:logistic",task="pred")
    dataAll <- xgb.DMatrix(data=data.matrix(train[,1:(ncol(train)-1)]), 
                           label=data.matrix(train[,ncol(train)]))
    
    res <- xgb.cv(params = param, data = dataAll,  nrounds = 1000, nfold = 10, 
                  prediction = TRUE, eval_metric="rmse", verbose = TRUE)
    max_round = which.min(res$dt[, test.rmse.mean])
    rmse <- res$dt[, test.rmse.mean][max_round]
    
    fit <- xgboost(params = param, data=dataAll, nrounds = max_round)
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
do_transform <- function(dataSet)
{
  for (i in 1:(ncol(dataSet) - 1) ) {
    if (min(dataSet[,i]) == 0 && max(dataSet[,i]) == 1 ) {
      #categorical values --> do nothing
    } else{
      dataSet[,i] <- log(dataSet[,i] + 1)
    }
  }
  dataSet[,ncol(dataSet)] <- log(dataSet[,ncol(dataSet)] + 1)
  return(dataSet)
}
reverse_transformed_response <- function(response)
{
  return(exp(response) - 1)
}
evaluate_model <- function(model, test)
{
  eval.predicted <- predict(model, data=test)
  eval.actual <- test$SalePrice
  eval.RMSE <- rmse(actual = eval.actual, predicted = eval.predicted)
  
}


#missing values: categorical + numeric values. Numeric values can be replaced by mean, median for ease.
#outliers: do later.
#do regression: variable selection + linear regression + xgboost + random forest
# check the residuals
# Mine: regression part + transformation 

####################### 1. Dealing with missing values for training and testing data #################################################
check_missing_values(HP.all)
HP.all <- deal_missing_values(HP.all, ntrain)
print ('check missing values of the dataframe after replacing missing values. Should print blank')
check_missing_values(HP.all)

#************************************************************************************************************************************#
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

#do transformation on variables?
do_transformation <- TRUE
if (do_transformation) {
  HP.all <- do_transform(HP.all)
}


HP.train = HP.all[1:ntrain,]
HP.test = HP.all[(ntrain+1) : nrow(HP.all),]
HP.test$SalePrice <- NULL

for (i in 1:ncol(HP.train)) {
  if (sum(is.nan(HP.train[,i])) > 0 || sum(is.infinite(HP.train[,i])) > 0 ) {
    print (i)
  }
}
for (i in 1:ncol(HP.test)) {
  if (sum(is.nan(HP.test[,i])) > 0 || sum(is.infinite(HP.test[,i])) > 0 ) {
    print (i)
  }
}

print ('Building multiple linear regression model')
HP.train.lr_model <- build_model(HP.train, type = 1, transformed = do_transformation)
print ('Building glmnet model')
HP.train.glmnet_model <- build_model(HP.train, type = 2, transformed = do_transformation)
print ('Building random forest model')
# HP.train.rf_model <- build_model(HP.train, type = 3, transformed = do_transformation)
print ('Building xgboost model')
HP.train.xgb_model <- build_model(HP.train, type = 4, transformed = do_transformation)

print('Choosing?')
HP.test$SalePrice <- predict(HP.train.xgb_model, as.matrix(HP.test))

if (do_transformation) HP.test$SalePrice <- reverse_transformed_response(HP.test$SalePrice)
#write to file
submission <- data.frame(Id <- HP.test.Id, SalePrice <- HP.test$SalePrice)
names(submission) <- c('Id', 'SalePrice')
write.csv(file = 'submission.csv', x = submission, row.names = FALSE)
