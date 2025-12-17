# ==============================================================================
# 0. CONFIGURACIÓN Y LIBRERÍAS
# ==============================================================================
if(!require(xgboost)) install.packages("xgboost")
if(!require(Matrix)) install.packages("Matrix")
if(!require(dplyr)) install.packages("dplyr")
if(!require(fastDummies)) install.packages("fastDummies")
if(!require(ParBayesianOptimization)) install.packages("ParBayesianOptimization")
if(!require(SHAPforxgboost)) install.packages("SHAPforxgboost")
if(!require(ggplot2)) install.packages("ggplot2")

library(xgboost)
library(Matrix)
library(dplyr)
library(fastDummies)
library(ParBayesianOptimization)
library(SHAPforxgboost)
library(ggplot2)

set.seed(123) # Reproducibilidad

# ==============================================================================
# 1. DEFINICIÓN DE FEATURES INICIALES (BASE)
# ==============================================================================
# Función base para generar features y poder analizar su importancia inicial
crear_features_base <- function(df) {
  df_out <- df %>%
    mutate(
      # Conversiones
      key = as.factor(key),
      time_signature = as.factor(time_signature),
      audio_mode = as.factor(audio_mode),
      
      # Interacciones Lógicas
      intensity = energy * (loudness + 60),
      dance_acoust_ratio = danceability / (acousticness + 0.01),
      mood_score = audio_valence * energy,
      
      # Logs para sesgo
      duration_log = log1p(song_duration_ms),
      liveness_log = log1p(liveness),
      is_instrumental = ifelse(instrumentalness > 0.05, 1, 0)
    )
  
  # Dummies
  df_out <- dummy_cols(df_out, select_columns = c("key", "time_signature"),
                       remove_first_dummy = FALSE, remove_selected_columns = TRUE)
  return(df_out)
}

train_base <- crear_features_base(train_imputed)

# Preparar matriz para el modelo exploratorio
cols_base <- setdiff(names(train_base), c("song_popularity", "ID"))
dtrain_base <- xgb.DMatrix(
  data = sparse.model.matrix(~ . -1, data = train_base[, cols_base]), 
  label = train_base$song_popularity # Target REAL (para RMSE)
)

# ==============================================================================
# 2. MODELO EXPLORATORIO Y ANÁLISIS DE GAIN (PODA)
# ==============================================================================

params_base <- list(booster = "gbtree", objective = "reg:squarederror", eta = 0.05, max_depth = 6)
model_base <- xgb.train(params = params_base, data = dtrain_base, nrounds = 100, verbose = 0)

# Obtener Importancia
imp <- xgb.importance(model = model_base)
imp <- imp[order(imp$Gain, decreasing = TRUE), ]

# Identificar las 15 peores
n_borrar <- 15
vars_borrar <- tail(imp$Feature, n_borrar)
cat("Variables a eliminar (Bottom 15 por Gain):\n")
print(vars_borrar)

# Definimos las columnas finales (Base - Borradas)
cols_filtradas <- setdiff(cols_base, vars_borrar)

# ==============================================================================
# 3. FEATURE ENGINEERING AVANZADO (CUADRÁTICAS + FILTRADO)
# ==============================================================================

preparar_dataset_final <- function(df, stats_ref = NULL) {
  df_out <- df %>%
    mutate(
      key = as.factor(key), time_signature = as.factor(time_signature), audio_mode = as.factor(audio_mode),
      intensity = energy * (loudness + 60),
      dance_acoust_ratio = danceability / (acousticness + 0.01),
      mood_score = audio_valence * energy,
      duration_log = log1p(song_duration_ms),
      liveness_log = log1p(liveness)
    )
  
  # Cuadráticas Centradas (Usando media del Train siempre)
  if (is.null(stats_ref)) {
    # MODO TRAIN: Calculamos medias
    media_tempo <- mean(df_out$tempo, na.rm = TRUE)
    media_intens <- mean(df_out$intensity, na.rm = TRUE)
    stats_out <- list(tempo = media_tempo, intensity = media_intens)
  } else {
    # MODO TEST: Usamos medias pasadas
    media_tempo <- stats_ref$tempo
    media_intens <- stats_ref$intensity
    stats_out <- stats_ref
  }
  
  df_out <- df_out %>%
    mutate(
      tempo_sq = (tempo - media_tempo)^2,
      intensity_sq = (intensity - media_intens)^2
    )
  
  df_out <- dummy_cols(df_out, select_columns = c("key", "time_signature", "audio_mode"),
                       remove_first_dummy = FALSE, remove_selected_columns = FALSE)
  
  # CORRECCIÓN DE NOMBRES (fastDummies a veces usa _0 y otras 0)
  colnames(df_out) <- gsub("audio_mode_0", "audio_mode0", colnames(df_out))
  colnames(df_out) <- gsub("audio_mode_1", "audio_mode1", colnames(df_out))
  
  return(list(data = df_out, stats = stats_out))
}


# Procesar Train
res_train <- preparar_dataset_final(train_imputed, stats_ref = NULL)
train_final <- res_train$data
stats_calculadas <- res_train$stats

# Procesar Test
res_test <- preparar_dataset_final(test_imputed, stats_ref = stats_calculadas)
test_final <- res_test$data

# ALINEACIÓN FINAL DE COLUMNAS (Train vs Test vs Filtradas)
# Nos quedamos solo con las columnas que sobrevivieron a la poda + las nuevas cuadráticas
cols_disponibles <- names(train_final)
# Agregamos las cuadráticas a la lista de permitidas si no estaban
cols_modelo_final <- unique(c(cols_filtradas, "tempo_sq", "intensity_sq"))
# Intersección para asegurar que existen
cols_modelo_final <- intersect(cols_modelo_final, cols_disponibles)

# Preparar Test: Rellenar faltantes con 0
for(col in cols_modelo_final) {
  if(!col %in% names(test_final)) test_final[[col]] <- 0
}

# MATRICES FINALES
dtrain_opt <- xgb.DMatrix(data = sparse.model.matrix(~ . -1, data = train_final[, cols_modelo_final]), 
                          label = train_final$song_popularity)
dtest_opt  <- xgb.DMatrix(data = sparse.model.matrix(~ . -1, data = test_final[, cols_modelo_final]))

cat(paste("Variables finales en el modelo:", length(cols_modelo_final), "\n"))

# ==============================================================================
# 4. OPTIMIZACIÓN BAYESIANA DE HIPERPARÁMETROS
# ==============================================================================

obj_func_rmse <- function(eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma) {
  params <- list(
    booster = "gbtree", objective = "reg:squarederror", eval_metric = "rmse",
    eta = eta, max_depth = as.integer(max_depth), min_child_weight = as.integer(min_child_weight),
    subsample = subsample, colsample_bytree = colsample_bytree, gamma = gamma
  )
  cv_res <- xgb.cv(params = params, data = dtrain_opt, nrounds = 1000, nfold = 3, 
                   early_stopping_rounds = 30, verbose = 0)
  return(list(Score = -min(cv_res$evaluation_log$test_rmse_mean), Pred = 0))
}

bounds <- list(
  eta = c(0.005, 0.1), max_depth = c(3L, 8L), min_child_weight = c(1L, 10L),
  subsample = c(0.6, 1.0), colsample_bytree = c(0.6, 1.0), gamma = c(0, 5)
)

opt_res <- bayesOpt(FUN = obj_func_rmse, bounds = bounds, initPoints = 8, iters.n = 15, acq = "ucb")
best_params_raw <- getBestPars(opt_res)

# Construir lista de parámetros ganadores
final_params <- list(
  booster = "gbtree", objective = "reg:squarederror", eval_metric = "rmse",
  eta = best_params_raw$eta, max_depth = as.integer(best_params_raw$max_depth),
  min_child_weight = as.integer(best_params_raw$min_child_weight),
  subsample = best_params_raw$subsample, colsample_bytree = best_params_raw$colsample_bytree,
  gamma = best_params_raw$gamma
)

cat("\n Parámetros Optimizados:\n")
print(final_params)

# ==============================================================================
# 5. VALIDACIÓN ROBUSTA (10-FOLD CV)
# ==============================================================================
cv_final <- xgb.cv(
  params = final_params, data = dtrain_opt, nrounds = 5000, nfold = 10,
  early_stopping_rounds = 50, verbose = 0, print_every_n = 100
)

best_iter <- cv_final$best_iteration
final_rmse <- cv_final$evaluation_log$test_rmse_mean[best_iter]
cat(paste0("📊 RMSE FINAL (10-Fold): ", round(final_rmse, 4), "\n"))

# ==============================================================================
# 6. ENTRENAMIENTO FINAL Y PREDICCIÓN
# ==============================================================================
model_production <- xgb.train(params = final_params, data = dtrain_opt, nrounds = best_iter, verbose = 0)

# Predecir
preds <- predict(model_production, dtest_opt)
preds <- pmax(0, pmin(100, preds)) # Clipping 0-100

# Guardar Submission
filename <- paste0("submission_xgb_master_", round(final_rmse, 4), ".csv")
submission <- data.frame(ID = test_final$ID, song_popularity = preds)
write.csv(submission, filename, row.names = FALSE)
cat(paste("Archivo generado:", filename, "\n"))

# ==============================================================================
# 7. INTERPRETACIÓN (SHAP & PLOTS)
# ==============================================================================

# Muestra para SHAP (Convertir matriz dispersa a densa para el plot)
set.seed(123)
X_sample_sparse <- sparse.model.matrix(~ . -1, data = train_final[, cols_modelo_final])
idx_sample <- sample(nrow(X_sample_sparse), min(2000, nrow(X_sample_sparse)))
X_sample_dense <- as.matrix(X_sample_sparse[idx_sample, ])

# Calcular SHAP
shap_vals <- shap.values(xgb_model = model_production, X_train = X_sample_dense)
shap_long <- shap.prep(shap_contrib = shap_vals$shap_score, X_train = X_sample_dense)

# Plot Beeswarm
p1 <- shap.plot.summary(shap_long) + ggtitle("Impacto de Variables (SHAP)")
print(p1)

# Plot Dependencia Cuadrática (Tempo)
p2 <- shap.plot.dependence(data_long = shap_long, x = "tempo_sq", y = "tempo_sq") + 
  ggtitle("Efecto no lineal del Tempo")
print(p2)

