# -----------------------------------------------------------------------------------------------------------------
# Setup e Caricamento Pacchetti
# -----------------------------------------------------------------------------------------------------------------

# Installa e carica i pacchetti necessari se non sono già installati
if (!requireNamespace("VIM", quietly = TRUE)) install.packages("VIM")
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
if (!requireNamespace("Metrics", quietly = TRUE)) install.packages("Metrics")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

library(VIM)
library(caret)
library(Metrics)
library(dplyr)


# -----------------------------------------------------------------------------------------------------------------
# Caricamento Dati e Calcolo dei Pesi
# -----------------------------------------------------------------------------------------------------------------

# **ATTENZIONE:** DEVI DEFINIRE path_intermediate e target_col nel tuo ambiente
# Esempio: 
# path_intermediate <- "C:/Tuo/Percorso/Ai/Dati"
# target_col <- "song_popularity"

# Variabile Target (DEVE CORRISPONDERE AL NOME DELLA COLONNA NEL TUO FILE CSV)
target_col <- "song_popularity"

# Caricamento del dataset
data_knn_vim <- read.csv(file.path(path_intermediate, "train_knn_vim.csv"))


### Calcolo dei Pesi (W1 e W2)
soglia_bassa <- 25

data_knn_vim$bassa_popolarita <- data_knn_vim[[target_col]] <= soglia_bassa
x <- sum(data_knn_vim$bassa_popolarita) # Numero di canzoni con popolarità bassa
y <- nrow(data_knn_vim) - x               # Numero di canzoni con popolarità alta

# Calcolo di W1 e W2 in modo che 70% del peso totale sia per la classe "bassa"
W1 <- 0.70 / x
W2 <- 0.30 / y

# Assegnazione della colonna pesi (case_weights)
data_knn_vim$case_weights <- 0
data_knn_vim$case_weights[data_knn_vim$bassa_popolarita] <- W1
data_knn_vim$case_weights[!data_knn_vim$bassa_popolarita] <- W2

# Rimuovi la colonna temporanea 'bassa_popolarita'
data_knn_vim <- data_knn_vim %>% dplyr::select(-bassa_popolarita)

cat("Pesi calcolati: W1 (bassa pop.) =", round(W1, 5), "W2 (alta pop.) =", round(W2, 5), "\n")
cat("Verifica somma pesi:", sum(data_knn_vim$case_weights), "\n\n")


# -----------------------------------------------------------------------------------------------------------------
# Funzioni kNN Ponderate Personalizzate
# -----------------------------------------------------------------------------------------------------------------

### Funzione di Previsione kNN Ponderata Personalizzata (Sostituisce VIM::kNN)
knn_vim_pred_weighted <- function(train, test, k, target) {
  
  # Estrai i pesi dal set di training
  train_weights <- train$case_weights
  
  # Rimuovi i pesi e il target dai dati di training e test per calcolare la distanza
  # L'assunto è che tutte le altre colonne sono features numeriche.
  train_features <- train %>% dplyr::select(-all_of(target), -case_weights)
  test_features <- test %>% dplyr::select(-all_of(target), -case_weights)
  
  preds <- numeric(nrow(test))
  
  # Loop sui punti di test
  for (i in 1:nrow(test)) {
    # 1. Calcolo Distanze (distanza Euclidea standard)
    distances <- as.matrix(dist(rbind(test_features[i, ], train_features)))[1, -1]
    
    # 2. Identificazione k Vicini
    nearest_neighbors_indices <- order(distances)[1:k]
    
    # 3. Estrazione Valori Target e Pesi dei Vicini
    neighbor_targets <- train[[target]][nearest_neighbors_indices]
    neighbor_weights <- train_weights[nearest_neighbors_indices]
    
    # 4. Previsione: Media Ponderata (usando i pesi W1/W2)
    preds[i] <- sum(neighbor_targets * neighbor_weights) / sum(neighbor_weights)
  }
  
  return(preds)
}


### Funzione di Test Principale (Aggiornata per usare il kNN ponderato)
knn_vim_test <- function(data, cv_test_folds, cv_tuning_folds,
                         target = target_col, tuning_search_space_k = c(3, 9, 27, 81, 100, 111)) {
  
  # Rimuovi la colonna 'case_weights' dal set target_vals per la CV
  data_no_weights <- data %>% dplyr::select(-case_weights)
  target_vals <- data_no_weights[, target]
  
  outer_folds <- createFolds(target_vals, k = cv_test_folds,
                             list = TRUE, returnTrain = TRUE)
  
  test_results <- data.frame(OuterFold = numeric(0), k = numeric(0), RMSE = numeric(0))
  overall_tuning_res <- data.frame(OuterFold = numeric(0), k = numeric(0), RMSE = numeric(0))
  
  # Outer Loop for Testing
  for (i in seq_along(outer_folds)) {
    cat("Outer Fold:", i, "\n")
    
    train_indices <- outer_folds[[i]]
    train_data <- data[train_indices, ]
    test_data <- data[-train_indices, ]
    
    tuning_res <- data.frame(OuterFold = i, k = tuning_search_space_k, RMSE = NA)
    
    # Inner Loop: CV for tuning
    for (l in seq_along(tuning_search_space_k)) {
      k <- tuning_search_space_k[[l]]
      rmse_k <- numeric(cv_tuning_folds)
      
      inner_folds <- createFolds(train_data[, target], k = cv_tuning_folds,
                                 list = TRUE, returnTrain = TRUE)
      
      for (j in seq_along(inner_folds)) {
        train_tune_indices <- inner_folds[[j]]
        train_tune_data <- train_data[train_tune_indices, ]
        validation_data <- train_data[-train_tune_indices, ]
        
        # Previsione con la funzione PERSONALIZZATA
        preds <- knn_vim_pred_weighted(train_tune_data, validation_data,
                                       k = k, target = target)
        
        rmse_k[j] <- rmse(validation_data[[target]], preds)
      }
      
      tuning_res[l, "RMSE"] <- mean(rmse_k)
    }
    
    best_k <- tuning_res$k[which.min(tuning_res$RMSE)]
    cat("Tuning suggested k =", best_k, "\n")
    
    overall_tuning_res <- rbind(overall_tuning_res, tuning_res)
    
    # Test il modello con il best k (funzione PERSONALIZZATA)
    preds_test <- knn_vim_pred_weighted(train_data, test_data,
                                        k = best_k, target = target)
    
    rmse_val <- rmse(test_data[, target], preds_test)
    
    test_results <- rbind(test_results,
                          data.frame(OuterFold = i, k = best_k, RMSE = rmse_val))
    
    cat("Test result using k =", best_k, ": RMSE =", round(rmse_val, 3), "\n")
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


# -----------------------------------------------------------------------------------------------------------------
# Esecuzione del Test
# -----------------------------------------------------------------------------------------------------------------
set.seed(123)

# La variabile target_col deve essere definita prima di questa esecuzione
results <- knn_vim_test(data_knn_vim, cv_test_folds = 3, cv_tuning_folds = 10, target = target_col)

## Visualizzazione dei risultati
cat("\n--- Risultati Tuning ---\n")
print(results$tuning_results)
cat("\n--- Risultati Test Per Fold ---\n")
print(results$test_results)
cat("\n--- Riassunto Performance Finale ---\n")
print(results$test_summary)