source("settings.R")
source("FE_v2[1]")
library(tictoc)

# Cross-Validation Control (used for all models)
ctrl <- trainControl(
  method = "cv", 
  number = 5, 
  verboseIter = TRUE
)

# Standard Grid Search for Radial SVM
grid_radial <- expand.grid(
  C = 10^(0:2),      # 1, 10, 100
  sigma = 10^(-3:-1) # 0.001, 0.01, 0.1
)

# Load Datasets
imputed_train <- read.csv(file.path(path_intermediate, "train_imputed.csv"))
imputed_test  <- read.csv(file.path(path_intermediate, "test_imputed.csv"))

# Subsets for simpler models
df_svm1 <- read.csv(file.path(path_intermediate, "train_svm1.csv")) # Discards unimportant vars
df_svm2 <- read.csv(file.path(path_intermediate, "train_svm2.csv")) # Keeps top 4 predictors

# Complete dataset with all the transformed variables (df from "Feature_Engineering")
#Convert all the variables in numeric
variabili_non_numeriche <- df %>%
  select_if(~!is.numeric(.)) %>%
  colnames()

if (length(variabili_non_numeriche) > 0) {
  # a. Ciclo per convertire le variabili character/factor in factor se non lo sono già
  for (col in variabili_non_numeriche) {
      df[[col]] <- as.numeric(df[[col]])
  }}





# This function handles the repetitive tasks: training, predicting, and printing errors.

run_svm_model <- function(train_data, test_data, method = "svmRadial", tune_grid = NULL, weights = NULL, label = "Model") {
  
  cat(paste0("\n------------------------------------------------------------\n"))
  cat(paste0("Training: ", label, "\n"))
  cat(paste0("------------------------------------------------------------\n"))
  
  set.seed(1234)
  
  # Set up training arguments
  train_args <- list(
    form = song_popularity ~ .,
    data = train_data,
    method = method,
    preProcess = c("center", "scale"),
    trControl = ctrl,
    tuneGrid = tune_grid
  )
  
  # Add weights if provided
  if (!is.null(weights)) {
    train_args$weights <- weights
  }
  
  # Train
  model <- do.call(train, train_args)
  print(model$bestTune)
  
  # Predict
  preds <- predict(model, newdata = test_data)
  
  # Evaluate
  rmse_val <- caret::RMSE(preds, test_data$song_popularity)
  mape_val <- tryCatch(mape(preds, test_data$song_popularity), error = function(e) NA)
  
  cat(paste("RMSE:", round(rmse_val, 4), "\n"))
  if(!is.na(mape_val)) cat(paste("MAPE:", round(mape_val, 4), "\n"))
  
  return(list(model = model, predictions = preds, rmse = rmse_val))
}

# ==============================================================================
# 4. MODELING
# ==============================================================================



# ------------------------------------------------------------------------------
# 4.1 FULL MODEL (Imputed Train)
# ------------------------------------------------------------------------------
# Sampling (Using your specific method)
set.seed(1234)
test_idx <- sample(1:nrow(imputed_train), size = nrow(imputed_train)/3)
dataTrain <- imputed_train[-test_idx, ]
dataTest  <- imputed_train[test_idx, ]

# Run Model
res_full <- run_svm_model(
  dataTrain, dataTest, 
  method = "svmRadial", 
  tune_grid = grid_radial, 
  label = "Full SVM (Radial)"
)

# View detailed results for the full model
print(res_full$model$results %>% 
        arrange(RMSE) %>% 
        dplyr::select(sigma, C, RMSE, Rsquared))

# Final Submission Generation (Using Full Model)
preds_final_test <- predict(res_full$model, newdata = imputed_test)
submission_svm <- data.frame(id = 1:length(preds_final_test), song_popularity = preds_final_test)
write.csv(submission_svm, file = file.path(path_intermediate, "submission_svm_1.csv"), row.names = FALSE)

# MAPE: 0.3008


# ------------------------------------------------------------------------------
# 4.2 MODEL SVM1 (Subset)
# ------------------------------------------------------------------------------
# Apply same sampling indices to ensure fair comparison
dataTrain_svm1 <- df_svm1[-test_idx, ]
dataTest_svm1  <- df_svm1[test_idx, ]

res_svm1 <- run_svm_model(
  dataTrain_svm1, dataTest_svm1, 
  method = "svmRadial", 
  tune_grid = grid_radial, 
  label = "SVM1 (Subset)"
)

# RMSE: 22.0397


# ------------------------------------------------------------------------------
# 4.3 MODEL SVM2 (Top 4 Predictors)
# ------------------------------------------------------------------------------
dataTrain_svm2 <- df_svm2[-test_idx, ]
dataTest_svm2  <- df_svm2[test_idx, ]

res_svm2 <- run_svm_model(
  dataTrain_svm2, dataTest_svm2, 
  method = "svmRadial", 
  tune_grid = grid_radial, 
  label = "SVM2 (Top 4 Vars)"
)

# The best model is the one with all the variables 


# ------------------------------------------------------------------------------
# 4.4 KERNEL EXPLORATION
# ------------------------------------------------------------------------------

# Linear Kernel
res_linear <- run_svm_model(
  dataTrain, dataTest, 
  method = "svmLinear", 
  tune_grid = expand.grid(C = c(0.1, 1, 10)), 
  label = "SVM Linear"
)

# Polynomial Kernel (using SVM1 data)
res_poly <- run_svm_model(
  dataTrain_svm1, dataTest_svm1, 
  method = "svmPoly", 
  tune_grid = expand.grid(C = c(0.5, 1.5), degree = c(2, 3), scale = c(0.5, 1.5)), 
  label = "SVM Polynomial"
)

#RMSE: 23.1954
#MAPE: 0.3384


# Create weights: 0.5 for outliers, 1.0 otherwise
train_with_outliers <- train_with_outliers %>%
  mutate(weights = ifelse(is_outlier == TRUE, 0.5, 1.0))

# Extract weights for the training set only
weights_train <- train_with_outliers[-test_idx_fe, ]$weights

res_weighted <- run_svm_model(
  dataTrain_fe, dataTest_fe, 
  weights = weights_train,
  tune_grid = expand.grid(C = 0.009, sigma = 0.4), 
  label = "Weighted SVM"
)


# ------------------------------------------------------------------------------
# 4.7 FEATURE SELECTION (RFE)
# ------------------------------------------------------------------------------

ctrl_rfe <- rfeControl(
  functions = caretFuncs, 
  method = "cv", 
  number = 5
)

# Exclude ID and Target columns for RFE input
# Note: Adjust column indices c(1, 15) if your dataset structure changes
rfe_x <- df_all_vbs[-test_idx_fe, -c(1, 15)]
rfe_y <- df_all_vbs[-test_idx_fe, ]$song_popularity

cat("Running Recursive Feature Elimination...\n")
results_rfe <- rfe(
  x = rfe_x, 
  y = rfe_y, 
  sizes = c(1:5, 10, 20, 47), 
  rfeControl = ctrl_rfe,
  method = "svmRadial", 
  tuneGrid = expand.grid(C = 1, sigma = 0.1)
)

# Select variables
vars_selected <- predictors(results_rfe)
cat("Selected Variables:", paste(vars_selected, collapse=", "), "\n")

# Create dataset with only selected features
df_selected <- df_all_vbs[, c(vars_selected, "song_popularity")]

# Split again for final validation
dataTrain_sel <- df_selected[-test_idx_fe, ]
dataTest_sel  <- df_selected[test_idx_fe, ]

res_rfe_final <- run_svm_model(
  dataTrain_sel, dataTest_sel,
  method = "svmRadial",
  tune_grid = expand.grid(C = 0.009, sigma = 0.4),
  label = "SVM after RFE"
)

# still rmse too high











#------------------------------------------------------------------------
# Feature selection
#------------------------------------------------------------------------

# Feature selection from the dataset used for XGBoost did not give a good RMSE
# Try feature selection with all the transformed variables 


controllo_rfe <- rfeControl(
  functions = caret::caretFuncs, 
  method = "cv", 
  number = 5 # Cross-validation a 5 fold
)

sizes <- c(1:5, 10, 20, 47)

# Esecuzione dell'RFE
risultato_rfe <- rfe(
  x = dataTrain_AllVbs[, - c(1, 15)], 
  y = dataTrain_AllVbs$song_popularity,  
  sizes = sizes, 
  rfeControl = controllo_rfe,
  method = "svmRadial", 
  tuneGrid = expand.grid(C = 1, sigma = 0.1))

# Visualizza le variabili selezionate
variabili_selezionate <- predictors(risultato_rfe)
names(imputed_train)
df_feature_selection <- df[, c(variabili_selezionate, "song_popularity")]

set.seed(1234)
test<-sample(1:nrow(df_feature_selection),size = nrow(df_feature_selection)/3)
dataTrain_fe<-df_feature_selection[-test,]
dataTest_fe<-df_feature_selection[test,]

model_svm_feat_sel <- train(
  song_popularity ~ .,
  data = dataTrain_fe,
  method = 'svmRadial',
  preProcess = c("center", "scale"),
  trControl = ctrl,     # Use the CV control defined above
  tuneGrid = expand.grid(C = 0.009, sigma = 0.4) # Use the grid search space
)

predictions_dataTest <- predict(model_svm_feat_sel, newdata = dataTest_fe)
rmse_dataTest <- RMSE(predictions_dataTest, dataTest$song_popularity)
cat(paste("RMSE on dataTest using best caret SVM model:", round(rmse_dataTest, 4), "\n\n"))
# RMSE still too high