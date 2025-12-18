source("settings.R")
library(finetune)
library(doParallel)

train_df <- read_csv(file.path(path_intermediate, "train_imputed.csv"))
test_df <- read_csv(file.path(path_intermediate, "test_imputed.csv"))


# Recipes ---------------------------------------------------------------------------------------------------------
recipe_1 <- recipe(song_popularity ~ ., data = train_df) %>%
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

recipe_2 <- recipe(song_popularity ~ ., data = train_df) %>%
  update_role(ID, new_role = "id_variable") %>%
  # Factor Transformation of categorical variables
  step_mutate(
    key = factor(key),
    audio_mode = factor(audio_mode),
    time_signature = factor(time_signature)
  ) %>%
  # Preprocessing
  step_corr(all_numeric_predictors(), threshold = 0.9) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)


recipe_3 <- recipe(song_popularity ~ ., data = train_df) %>%
  update_role(ID, new_role = "id_variable") %>%
  # Factor Transformation of categorical variables
  step_mutate(
    key = factor(key),
    time_signature = factor(time_signature),
    audio_mode = factor(audio_mode, labels = c("minor", "major"))
  ) %>%
  # log1p or other Transformations for certain variables
  step_mutate(
    has_instrumental = ifelse(instrumentalness > 0, 1, 0),
    instrumentalness = log1p(instrumentalness),
    acousticness = log1p(acousticness),
    liveness = log1p(liveness),
    speechiness = log1p(speechiness),
    tempo = pmin(tempo, quantile(tempo, 0.99))
  ) %>%
  # Preprocessing
  step_corr(all_numeric_predictors(), threshold = 0.9) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)


# RF Model Definition ---------------------------------------------------------------------------------------------
rf_spec <- rand_forest(
  mtry = tune(),        # Número de predictores muestreados
  min_n = tune(),       # Tamaño mínimo del nodo para dividir
  trees = 500           # Número de árboles
) %>%
  set_mode("regression") %>%
  set_engine("ranger")


# Grid ------------------------------------------------------------------------------------------------------------
# Anzahl Features NACH dummy-encoding
num_features <- function(recipe) {
  ncol(juice(prep(recipe, training = train_df))) - 1  # -1 wegen outcome
}

# rf_grid <- function(recipe) {
#   n_feat <- num_features(recipe)
#   grid_regular(
#     mtry(range = c(3, floor(sqrt(n_feat)))),
#     min_n(range = c(2, 20)),
#     levels = 10
#   )
# }

rf_grid <- function(recipe) {
  n_feat <- num_features(recipe)
  grid_random(
    mtry(range = c(3, floor(n_feat / 2))),
    min_n(range = c(2, 20)),
    size = 50
  )
}


# work flow -------------------------------------------------------------------------------------------------------
rf_workflow <- function(recipe) {
  workflow() %>%
    add_recipe(recipe) %>%
    add_model(rf_spec)
}

# Tuning --------------------------------------------------------------------------------------------------------------
set.seed(42)
cv_folds <- vfold_cv(train_df, v = 10, strata = song_popularity)

# rf_tune_results <- tune_grid(
#   rf_workflow,
#   resamples = cv_folds,
#   grid = rf_grid,
#   metrics = metric_set(rmse, mae, rsq), 
#   control = control_grid(verbose = TRUE)
# )

race_ctrl <- control_race(
  verbose = TRUE,
  verbose_elim = TRUE,
  save_pred = FALSE,
  parallel_over = "resamples",
  burn_in = 4
)


rf_tune_results <- function(recipe) {
  rf_wf <- rf_workflow(recipe)
  rf_grd <- rf_grid(recipe)
  tune_race_anova(
    rf_wf,
    resamples = cv_folds,
    grid = rf_grd,
    metrics = metric_set(rmse),
    control = race_ctrl
  )
}


# Evaluating Recipes ----------------------------------------------------------------------------------------------
recipe_list <- list(recipe_1, recipe_2, recipe_3)

# use 3 Worker on a 4-kernel-CPU
cl <- makePSOCKcluster(3)
registerDoParallel(cl)

for (i in seq_along(recipe_list)) {
  if (!file.exists(paste0("Results/RandomForest/rf_tuning_rec_", i, ".rds"))) {
    res <- rf_tune_results(recipe_list[[i]])
    saveRDS(res, file = paste0("Results/RandomForest/rf_tuning_rec_", i, ".rds"))
  }
}

stopCluster(cl)
registerDoSEQ()



## -> Best Model with recipe 1
best_recipe <- 1



# Tune Model with recipe 1 ----------------------------------------------------------------------------------------
rf_spec <- rand_forest(
  mtry = tune(),        # Número de predictores muestreados
  min_n = tune(),       # Tamaño mínimo del nodo para dividir
  trees = 1000           # Número de árboles
) %>%
  set_mode("regression") %>%
  set_engine("ranger")


rf_grid <- function(recipe) {
  grid_random(
    mtry(range = c(3, 15)),
    min_n(range = c(2, 10)),
    size = 75
  )
}

set.seed(123)
cv_folds <- vfold_cv(train_df, v = 10, strata = song_popularity)

race_ctrl <- control_race(
  verbose = TRUE,
  verbose_elim = TRUE,
  save_pred = FALSE,
  parallel_over = "resamples",
  burn_in = 3
)


# use 3 Worker on a 4-kernel-CPU
cl <- makePSOCKcluster(3)
registerDoParallel(cl)

if (!file.exists("Results/RandomForest/rf_tuning_final.rds")) {
  res <- rf_tune_results(recipe_list[[best_recipe]])
  saveRDS(res, file = "Results/RandomForest/rf_tuning_final.rds")
}

stopCluster(cl)
registerDoSEQ()


# Finalize Workflow with best recipe and parameters ---------------------------------------------------------------
rf_tune_1 <- readRDS("Results/rf_tuning_final.rds")
best_params <- select_best(rf_tune_1, metric = "rmse")
print(best_params)

final_rf_workflow <- rf_workflow(recipe_1) %>%
  finalize_workflow(best_params)

# Final Fit
final_rf_model <- fit(final_rf_workflow, data = train_df)

# Test set Prediction
test_predictions <- predict(final_rf_model, new_data = test_df)

# Submission Format
submission_df <- test_df %>%
  dplyr::select(ID) %>%
  bind_cols(test_predictions) %>%
  rename(song_popularity = .pred)

write_csv(submission_df, "Results/RandomForest/submission_rf_fe_1.csv")
