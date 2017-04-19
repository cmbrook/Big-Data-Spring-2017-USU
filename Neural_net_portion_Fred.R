
train_dum<-encode_categorical_variables(train2)
ntrain<-1460
train_dum_reduced<-do_variable_selection(train_dum)

vrs<-names(train_dum_reduced)

train_dum_reduced_log<-mutate(train_dum_reduced,SalePrice=log(SalePrice+1))


nn_form<-as.formula(paste("SalePrice ~", paste(vrs[!vrs %in% "SalePrice"],collapse= " + ")))



train_dum_scaled<-as.data.frame(scale(train_dum_reduced_log, center=T,scale=T))


nn_onenode<-neuralnet(nn_form,dat=train_dum_scaled, hidden=c(1),linear.output=T,rep=5,stepmax=1e+05,lifesign="full")


#doesn't converge with logistic activation function
#hnn_20nodes<-neuralnet(nn_form,data=train_dum_scaled,hidden=c(10,10),linear.output=T,rep=1,stepmax=3e+05,lifesign="full")

#tried with tanh activation function
#hnn_20nodes_t<-neuralnet(nn_form,data=train_dum_scaled,hidden=c(10,10),linear.output=T,rep=5,stepmax=5e+05,lifesign="full",act.fct = "tanh")

#hnn_20nodes_l_cv<-tenFoldCrossVal(formula=nn_form,data=train_dum_scaled,type="neunet",nnn=c(10,10))

#results<-hnn_20nodes_t$result.matrix


#this one works after 11 minutes
#hnn_25nodes<-neuralnet(nn_form,data=train_dum_scaled,hidden=c(15,10),linear.output=T,rep=1,stepmax=3e+05,lifesign="full")

# but cross validation doesn't work
#nn_cv_25node<-tenFoldCrossVal(formula=nn_form,data=train_dum_scaled,type="neunet",nnn=c(15,10))

#didn't converge
#hnn_25n4L<-neuralnet(nn_form,data=train_dum_scaled,hidden=c(12,8,4,2),linear.output=T,rep=1,stepmax=3e+05,lifesign="full")

#converged
#hhn_48n<-neuralnet(nn_form,data=train_dum_scaled,hidden=c(15),linear.output=T,rep=1,stepmax=3e+05,lifesign="full")

#but cross validation doesn't work
#hhn_15_cv<-tenFoldCrossVal(formula=nn_form,data=train_dum_scaled,type="neunet",nnn=c(15),output=T,stepmax=3e+05,lifesign="full")

#nn_15_prediction_train<-unscale_unlog_nn_prediction(compute(hhn_48n,covariate=select(train_dum_scaled,-SalePrice))$net.result,train_dum_reduced_log$SalePrice)

RMSE<-sqrt(sum((train_dum_reduced$SalePrice - nn_15_prediction_train)**2)/1460) 

#RMSE is extremely bad

#Code to predict from test data and create submission csv
test1<-deal_missing_values(test)
test_dum<-encode_categorical_variables(test1)
reduced_vars<-which(colnames(test_dum) %in% colnames(train_dum_reduced))

test_dum_reduced<-test_dum[,reduced_vars]

test_dum_scaled<-scale(test_dum_reduced)

prediction_nn_un<-compute(hhn_48n,covariate=test_dum_scaled)$net.result

prediction_nn<-unscale_unlog_nn_prediction(prediction_nn_un,train_dum_reduced_log$SalePrice)

prediction_nn_un_25t<-compute(hnn_20nodes_t,covariate=test_dum_scaled,rep=2)$net.result

prediction_nn_25t<-unscale_unlog_nn_prediction(prediction_nn_un_25t,train_dum_reduced_log$SalePrice)

submission_nn<-data.frame(test$Id,prediction_nn)
colnames(submission_nn)<-c("ID","SalePrice")

write.csv(submission_nn,"submission_nn_15.csv",row.names = FALSE)


# We tried a variety of configurations for a neural net using a subset of predictors, and found very few configurations that converged consistently. 
# 
# Even when the models converged with the entirety of the data, we could not achieve convergence for most of models using a 10-fold cross validation method. Since these models took an extremely long time to run and did not appear to have better predictive power than other methods, we decided not to use a neural net model for our final submission. The RMLSE for an untuned 1-node neural net model was .16521, worse than the linear model. 