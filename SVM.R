source("settings.R")
source("FE_v2[1]")
library(tictoc)

# Training dataset with all the original variables
imputed_train <- read.csv(file.path(path_intermediate, "train_imputed.csv"))
imputed_test <- read.csv(file.path(path_intermediate, "test_imputed.csv"))

# Dataset with preprocessing (simpler model)
# svm1 discard the variables that are not considered important predictors
# svm2 keeps only the 4 most important predictors according to EDA
df_svm1 <- read.csv(file.path(path_intermediate, "train_svm1.csv"))
df_svm2 <- read.csv(file.path(path_intermediate, "train_svm2.csv"))

# Feature engeneering used for XGBoost
df_fe <- preparar_dataset_final(imputed_train)$data
set.seed(1234)
test<-sample(1:nrow(df_fe),size = nrow(df_fe)/3)
dataTrain_fe<-df_fe[-test,]
dataTest_fe<-df_fe[test,]
  
#----------------------------------------------------------------------------
#FULL MODEL
#----------------------------------------------------------------------------

set.seed(1234)
test<-sample(1:nrow(imputed_train),size = nrow(imputed_train)/3)
dataTrain<-imputed_train[-test,]
dataTest<-imputed_train[test,]

#Tuning with caret
ctrl <- trainControl(
  method = "cv", # K-Fold Cross-Validation
  number = 5,    # Number of folds (5 is often a good balance of speed/robustness)
  verboseIter = TRUE # See progress
)

tuneGrid_svm <- expand.grid(
  C = 10^(0:2),        # Try C = 1, 10, 100
  sigma = 10^(-3:-1)    # Try sigma = 0.001, 0.01, 0.1
)


model_svm <- train(
  song_popularity ~ .,
  data = dataTrain,
  method = 'svmRadial',
  preProcess = c("center", "scale"),
  trControl = ctrl,     # Use the CV control defined above
  tuneGrid = tuneGrid_svm # Use the grid search space
)

print(model_svm$bestTune)

# Show the results table for all tested parameters (sorted by RMSE)
print(model_svm$results %>%
        arrange(RMSE) %>%
        dplyr::select(sigma, C, RMSE, Rsquared))

# plot(model_svm_tuned)

# Predict on the test set using the best model found during tuning
predictions_test <- predict(model_svm, newdata = dataTest)

# Calculate RMSE on the test set
test_rmse <- caret::RMSE(predictions_test, dataTest$song_popularity)
cat(paste("RMSE en el Conjunto de Prueba:", round(test_rmse, 4), "\n"))

# Calculate R-squared on the test set
test_r_squared <- R2(predictions_test, dataTest$song_popularity)
cat(paste("R-squared en el Conjunto de Prueba:", round(test_r_squared, 4), "\n"))

best_tuneGrid <- expand.grid(C = 1, sigma = 0.01)

# Final prediction (for submission)
# Predictions on the 'test' dataset (without song_popularity)
predictions_final_test <- predict(model_svm, newdata = imputed_test)
sample_submission_svm <- data.frame(id = 1:length(predictions_final_test), song_popularity = as.integer(predictions_final_test))


#------------------------------------------------------------------------------
#Model svm1

# keep all the variables except liveness, speechiness, tempo
#------------------------------------------------------------------------------

dataTrain_svm1<-df_svm1[-test,]
dataTest_svm1<-df_svm1[test,]

model_svm_1 <- train(
  song_popularity ~ .,
  data = dataTrain_svm1,
  method = 'svmRadial',
  preProcess = c("center", "scale"),
  trControl = ctrl,     # Use the CV control defined above
  tuneGrid = tuneGrid_svm # Use the grid search space
)

#print(model_svm_1)

# Predictions on dataTest
predictions_dataTest_svm1 <- predict(model_svm_1, newdata = dataTest_svm1)

# Calculate RMSE for dataTest
rmse_dataTest <- RMSE(predictions_dataTest_svm1, dataTest_svm1$song_popularity)
cat(paste("RMSE on dataTest using best caret SVM model:", round(rmse_dataTest, 4), "\n\n"))

# RMSE: 22.0397

#------------------------------------------------------------------------------
#Model svm2

# keep only danceability, energy, acousticness, instrumentalness.
#------------------------------------------------------------------------------

dataTrain_svm2<-df_svm2[-test,]
dataTest_svm2<-df_svm2[test,]

model_svm_2 <- train(
  song_popularity ~ .,
  data = dataTrain_svm2,
  method = 'svmRadial',
  preProcess = c("center", "scale"),
  trControl = ctrl,     # Use the CV control defined above
  tuneGrid = tuneGrid_svm # Use the grid search space
)

#print(model_svm_1)

# Predictions on dataTest
predictions_dataTest_svm2 <- predict(model_svm_2, newdata = dataTest_svm2)

# Calculate RMSE for dataTest

rmse_dataTest <- RMSE(predictions_dataTest_svm2, dataTest_svm1$song_popularity)
cat(paste("RMSE on dataTest using best caret SVM model:", round(rmse_dataTest, 4), "\n\n"))


#---------------------------------------------------------------------------
# The best model is the one with all the variables
# Now try other kernels
#---------------------------------------------------------------------------

# You can inspect the tuning parameters for any method like this:
modelLookup('svmLinear')
modelLookup('svmPoly')
modelLookup('svmSigmoid')
# And, of course, the one you used:
modelLookup('svmRadial')

###### Linear kernel

model_svm_linear <- train(
  song_popularity ~ .,
  data = dataTrain,
  method = 'svmLinear', # Changed to svmLinear
  preProcess = c("center", "scale"),
  trControl = ctrl,
  tuneGrid = expand.grid(C = c(0.1, 1, 10)) # Only C for linear kernel
)

predictions_dataTest_linear <- predict(model_svm_linear, newdata = dataTest)

# Calculate RMSE for dataTest

rmse_dataTest <- RMSE(predictions_dataTest_linear, dataTest$song_popularity)
cat(paste("RMSE on dataTest using best caret SVM model:", round(rmse_dataTest, 4), "\n\n"))



######## Polynomial kernel

model_svm_poly <- train(
  song_popularity ~ .,
  data = dataTrain_svm1,
  method = 'svmPoly', # Changed to svmPoly
  preProcess = c("center", "scale"),
  trControl = ctrl,
  tuneGrid = expand.grid(C = c(0.5, 1.5), degree = c(2, 3), scale = c(0.5,1.5))
)

predictions_dataTest_poly <- predict(model_svm_poly, newdata = dataTest_svm1)

# Calculate RMSE for dataTest

rmse_dataTest <- RMSE(predictions_dataTest_poly, dataTest$song_popularity)
cat(paste("RMSE on dataTest using best caret SVM model:", round(rmse_dataTest, 4), "\n\n"))
#RMSE: 23.1954

mape_dataTest <- mape(predictions_dataTest_poly, dataTest$song_popularity)
cat(paste("mape on dataTest using best caret SVM model:", round(mape_dataTest, 4), "\n\n"))
#MAPE: 0.3384

#------------------------------------------------------------------------
# Try new feature engeneering
#------------------------------------------------------------------------

tic()
model_svm_fe <- train(
  song_popularity ~ .,
  data = dataTrain_fe,
  method = 'svmRadial',
  preProcess = c("center", "scale"),
  trControl = ctrl,     # Use the CV control defined above
  tuneGrid = expand.grid(C = c(0.4, 0.1), sigma = c(0.009, 0.001)) # Use the grid search space
)
toc()
#selection: sigma=0.01, C=0.5

predictions_dataTest <- predict(model_svm_fe, newdata = dataTest_fe)

# Calculate RMSE for dataTest

rmse_dataTest <- RMSE(predictions_dataTest, dataTest_fe$song_popularity)
cat(paste("RMSE on dataTest using best caret SVM model:", round(rmse_dataTest, 4), "\n\n"))
#RMSE: 21.9962

mape_dataTest <- mape(predictions_dataTest, dataTest_fe$song_popularity)
cat(paste("mape on dataTest using best caret SVM model:", round(mape_dataTest, 4), "\n\n"))
#MAPE: 0.3079

#------------------------------------------------------------------------
# Put weights for outliers
#------------------------------------------------------------------------

train_with_outliers <- train_with_outliers %>%
  dplyr::mutate(
    weights = ifelse(is_outlier == TRUE, 
                     0.5, # Se è un outlier (TRUE), il peso è 0.5
                     1.0  # Altrimenti (FALSE), il peso è 1.0
    )
  )

weights_train <- train_with_outliers[-test,]
weights_train <- weights_train$weights

model_svm_pesato <- train(
  song_popularity ~ .,
  data = dataTrain_fe,
  method = 'svmRadial',
  weights = weights_train,
  preProcess = c("center", "scale"),
  trControl = ctrl,     # Use the CV control defined above
  tuneGrid = expand.grid(C = 0.009, sigma = 0.4) # Use the grid search space
)
predictions_dataTest <- predict(model_svm_pesato, newdata = dataTest)
rmse_dataTest <- RMSE(predictions_dataTest, dataTest$song_popularity)
cat(paste("RMSE on dataTest using best caret SVM model:", round(rmse_dataTest, 4), "\n\n"))


#------------------------------------------------------------------------
# Feature selection
#------------------------------------------------------------------------

# add feature selection with all the variables 

controllo_rfe <- rfeControl(
  functions = caret::caretFuncs, 
  method = "cv", 
  number = 5 # Cross-validation a 5 fold
)

sizes <- c(1:5, 10, 15, 20, ncol(dataTrain) - 1)

# Esecuzione dell'RFE
risultato_rfe <- rfe(
  x = dataTrain[, -2], # Predittori
  y = dataTrain$song_popularity,  # Variabile Target
  sizes = sizes, # Prova sottoinsiemi da 1 a 10 variabili
  rfeControl = controllo_rfe,
  method = "svmRadial", # Modello SVR per la selezione
  tuneGrid = expand.grid(C = 1, sigma = 0.01),
  preProcess = c("center", "scale")
)

# Visualizza le variabili selezionate
variabili_selezionate <- predictors(risultato_rfe)
names(imputed_train)
df_feature_selection <- imputed_train[, c(3, 13, 15)]

set.seed(1234)
test<-sample(1:nrow(df_fe),size = nrow(df_fe)/3)
dataTrain_fe<-df_feature_selection[-test,]
dataTest_fe<-df_feature_selection[test,]

model_svm_feat_sel <- train(
  song_popularity ~ .,
  data = dataTrain,
  method = 'svmRadial',
  preProcess = c("center", "scale"),
  trControl = ctrl,     # Use the CV control defined above
  tuneGrid = expand.grid(C = 0.009, sigma = 0.4) # Use the grid search space
)

predictions_dataTest <- predict(model_svm_feat_sel, newdata = dataTest)
rmse_dataTest <- RMSE(predictions_dataTest, dataTest$song_popularity)
cat(paste("RMSE on dataTest using best caret SVM model:", round(rmse_dataTest, 4), "\n\n"))
