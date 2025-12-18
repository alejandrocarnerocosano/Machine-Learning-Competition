source("settings.R")
library(finetune)
library(vip)
library(doParallel)


# RMSE and Mape Functions -------------------------------------------------------------------------------------------------
rmse_fn <- function(truth, pred) {
  sqrt(mean((pred - truth)^2))
}

mape_fn <- function(truth, pred) {
  mean(abs(pred - truth) / (truth + 1))
}


# Read Data and Tuning Results-------------------------------------------------------------------------------------------------------
train_df <- read_csv(file.path(path_intermediate, "train_imputed.csv"))
test_df <- read_csv(file.path(path_intermediate, "test_imputed.csv"))
tune_results <- readRDS("Results/RandomForest/rf_tuning_rec_1.rds")

best_params <- select_best(tune_results, metric = "rmse")


# Workflow Definition ---------------------------------------------------------------------------------------------
recipe_rf <- recipe(song_popularity ~ ., data = train_df) %>%
  update_role(ID, new_role = "id_variable") %>%
  # Factor Transformation of categorical variables
  step_mutate(
    key = factor(key),
    audio_mode = factor(audio_mode),
    time_signature = factor(time_signature)
  ) %>%
  # Preprocessing
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

rf_model <- rand_forest(
  trees = 5000,
  mtry = best_params$mtry,
  min_n = best_params$min_n
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

rf_workflow <- workflow() %>%
  add_recipe(recipe_rf) %>%
  add_model(rf_model)


# CV Test ---------------------------------------------------------------------------------------------------------
set.seed(123)

cv_folds <- vfold_cv(train_df, v = 3, strata = song_popularity)

if (!file.exists("Results/RandomForest/rf_test_cv_summary.rds")) {
  rmse_values <- c()
  mape_values <- c()
  
  for (i in seq_along(cv_folds$splits)) {
    
    # train and test set for the fold
    dtrain_fold <- analysis(cv_folds$splits[[i]])
    dtest_fold  <- assessment(cv_folds$splits[[i]])
    
    # fit model on train set of the fold
    rf_fit <- fit(rf_workflow, data = dtrain_fold)
    
    # predict on test set of the fold
    preds <- predict(rf_fit, as_tibble(dtest_fold))$.pred
    
    # calculate RMSE and MAPE
    rmse_values[i] <- rmse_fn(dtest_fold$song_popularity, preds)
    mape_values[i] <- mape_fn(dtest_fold$song_popularity, preds)
  }
  
  # Summary
  cv_summary <- tibble(
    rmse_mean = mean(rmse_values),
    rmse_sd   = sd(rmse_values),
    mape_mean = mean(mape_values),
    mape_sd   = sd(mape_values)
  )
  
  saveRDS(cv_summary, file = "Results/RandomForest/rf_test_cv_summary.rds")
  
}


# Final Fit -------------------------------------------------------------------------------------------------------
rf_model_final <- rand_forest(
  trees = 5000,
  mtry = best_params$mtry,
  min_n = best_params$min_n
) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("regression")

rf_final_workflow <- workflow() %>%
  add_recipe(recipe_rf) %>%
  add_model(rf_model_final)

rf_fit_final <- fit(rf_final_workflow, data = train_df)

# Feature Importance
rf_obj <- extract_fit_parsnip(rf_fit_final)$fit

importance_values <- rf_obj$variable.importance

importance_df <- data.frame(
  feature = names(importance_values),
  importance = importance_values
) %>%
  dplyr::arrange(desc(importance))

print(importance_df)

# Optional: Top 10 Features
head(importance_df, 10)

# Optional: Visualisierung
vip(rf_obj, geom = "col") + ggtitle("Permutation Feature Importance - Random Forest")



# Test Set Predictions ---------------------------------------------------------------------------------------------------------
test_predictions <- predict(rf_fit_final, new_data = test_df)


# Submission ------------------------------------------------------------------------------------------------------
submission_df <- test_df %>%
  dplyr::select(ID) %>%
  # Añade la columna de predicción
  bind_cols(test_predictions) %>%
  rename(song_popularity = .pred) #%>%

write_csv(submission_df, "Results/RandomForest/submission_rf_final.csv")


