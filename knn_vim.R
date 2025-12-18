source("settings.R")


# Functions -------------------------------------------------------------------------------------------------------
### Prediction Function
# train: full imputed data set
# test: set with no missing values in feature values and only NA in target column
knn_vim_pred <- function(train, test, k, target = "song_popularity") {
  test[, target] <- NA
  data <- rbind(train, test)
  test_row_ids <- seq(nrow(train) + 1, nrow(train) + nrow(test))
  result <- kNN(data, k=k, variable = target, weightDist = FALSE)
  return(result[test_row_ids, ])
}


### Test Function
## input
# data: full imputed data set
# cv_test_folds: number of folds of outer CV
# cv_tuning_folds: number of folds of inner CV
# target: column name in the data set corresponding to the target variable
# tuning_search_space_k: values of k to try/ tune for
knn_vim_test <- function(data, cv_test_folds, cv_tuning_folds,
                         target = "song_popularity", tuning_search_space_k = c(3, 9, 27, 81, 100, 111)) {
  target_vals <- data[, target]
  
  outer_folds <- createFolds(target_vals, k = cv_test_folds,
                             list = TRUE, returnTrain = TRUE)
  
  # Two empty data sets to store the results
  test_results <- data.frame(OuterFold = numeric(0),
                             k = numeric(0),
                             RMSE = numeric(0))
  
  overall_tuning_res <- data.frame(OuterFold = numeric(0),
                                   k = numeric(0),
                                   RMSE = numeric(0),
                                   MAE = numeric(0))
  
  # Outer Loop for Testing
  for (i in seq_along(outer_folds)) {
    
    cat("Outer Fold:", i, "\n")
    
    # Train/ Test Split
    train_indices <- outer_folds[[i]]
    train_data <- data[train_indices, ]
    test_data <- data[-train_indices, ]
    
    inner_folds <- createFolds(train_data[, target], k = cv_tuning_folds,
                               list = TRUE, returnTrain = TRUE)
    
    tuning_res <- data.frame(OuterFold = i,
                             k = tuning_search_space_k,
                             RMSE = NA,
                             MAE = NA)
    
    # Inner Loop: cv_tuning_folds-CV for tuning
    for (l in seq_along(tuning_search_space_k)) {
      k <- tuning_search_space_k[[l]]
      rmse_k <- numeric(length(inner_folds))
      mae_k <- numeric(length(inner_folds))
      
      for (j in seq_along(inner_folds)) {
        # Tuning Train/ Validation Split
        train_tune_indices <- inner_folds[[j]]
        train_tune_data <- train_data[train_tune_indices, ]
        validation_data <- train_data[-train_tune_indices, ]
        
        preds <- knn_vim_pred(train_tune_data, validation_data,
                              k = k, target = target)[, target]
        
        rmse_k[j] <- rmse(validation_data[[target]], preds)
        mae_k[j] <- mae(validation_data[[target]], preds)
      }
      
      tuning_res[l, "RMSE"] <- mean(rmse_k)
      tuning_res[l, "MAE"] <- mean(mae_k)
    }
    
    best_k <- tuning_res$k[which.min(tuning_res$RMSE)]
    
    cat("Tuning suggested k = ", best_k, "\n")
    
    # Save tuning results
    overall_tuning_res <- rbind(overall_tuning_res, tuning_res)
    
    # Learn the model on the outer train set with the best k
    # and test it using the outer test set
    preds_test <- knn_vim_pred(train_data, test_data,
                               k = best_k, target = target)[, target]
    
    # Performance estimation
    rmse_val <- rmse(test_data[, target], preds_test)
    
    # Save test result
    test_results <- rbind(test_results,
                          data.frame(OuterFold = i,
                                     k = best_k,
                                     RMSE = rmse_val))
    
    cat("Test result using k = ", best_k, ": RMSE = ", round(rmse_val, 3), "\n")
  }
  
  summary_test_results <- test_results %>%
    summarise(mean_RMSE = mean(RMSE),
              sd_RMSE = sd(RMSE))
  
  return(list(
    tuning_results = overall_tuning_res,
    test_results = test_results,
    test_summary = summary_test_results
  ))
}



# Perform Test ----------------------------------------------------------------------------------------------------
set.seed(123)

data_knn_vim <- read.csv(file.path(path_intermediate, "train_knn_vim.csv"))

results <- knn_vim_test(data_knn_vim, cv_test_folds = 3, cv_tuning_folds = 10)

results$tuning_results
results$test_results
results$test_summary



# Save Test Results -----------------------------------------------------------------------------------------------

saveRDS(results$tuning_results, file = "Results/KNN/knn_vim_tuning.rds")
saveRDS(results$test_results, file = "Results/KNN/knn_vim_test.rds")
saveRDS(results$test_summary, file = "Results/KNN/knn_vim_test_summary.rds")

