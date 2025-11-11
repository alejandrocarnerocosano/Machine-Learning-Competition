#----------------------------------------------------------------#
# 1. INSTALACIÓN Y CARGA DE LIBRERÍAS
#----------------------------------------------------------------#

# Instalar librerías si no están presentes
if (!require("solitude")) install.packages("solitude")
if (!require("Rlof")) install.packages("Rlof")
if (!require("tidyverse")) install.packages("tidyverse")

# Cargar librerías
library(solitude)    # Para Isolation Forest
library(Rlof)        # Para Local Outlier Factor (LOF)
library(tidyverse)   # Para manipulación de datos

#----------------------------------------------------------------#
# 2. CREACIÓN DE DATAFRAME DE EJEMPLO 'train'
#----------------------------------------------------------------#

train <- read.csv("~/Downloads/imputedn.csv")


cat("Dimensiones del dataframe 'train':", dim(train), "\n")
print(tail(train)) # Ver los outliers añadidos al final

#----------------------------------------------------------------#
# 3. APLICACIÓN DE MÉTODOS DE DETECCIÓN
#----------------------------------------------------------------#

# --- MÉTODO 1: ISOLATION FOREST (RECOMENDADO) ---

# Crear un objeto isolationForest. El documento usa sample_size y num_trees[cite: 46].
isoforest <- isolationForest$new(sample_size = 64, num_trees = 100)

# Ajustar el modelo a los datos
isoforest$fit(train)

# Predecir las puntuaciones de anomalía.
# Puntuaciones más bajas en 'average_depth' o más altas en 'anomaly_score' indican outliers.
predicciones_iso <- isoforest$predict(train)

# Identificar outliers usando un cuantil. Por ejemplo, el 5% más anómalo.
umbral_iso <- quantile(predicciones_iso$anomaly_score, 0.999)
outliers_iso_indices <- which(predicciones_iso$anomaly_score >= umbral_iso)
print(length(outliers_iso_indices))
length(outliers_iso_indices)/nrow(train)
cat("\n--- Isolation Forest ---\n")
cat("Umbral de puntuación de anomalía (percentil 95):", umbral_iso, "\n")
cat("Índices de Outliers detectados:", outliers_iso_indices, "\n")
print(train[outliers_iso_indices, ])


# --- MÉTODO 2: LOCAL OUTLIER FACTOR (LOF) ---

# Calcular las puntuaciones LOF. 'k' es el número de vecinos.
# Un valor LOF alto indica un outlier.
lof_scores <- lof(train, k = 5)

# Identificar outliers. Podemos tomar los N con mayor puntuación.
umbral_lof <- quantile(lof_scores, 0.95)
outliers_lof_indices <- which(lof_scores >= umbral_lof)

cat("\n--- Local Outlier Factor (LOF) ---\n")
cat("Umbral de puntuación LOF (percentil 95):", umbral_lof, "\n")
cat("Índices de Outliers detectados:", outliers_lof_indices, "\n")
print(train[outliers_lof_indices, ])


# --- MÉTODO 3: COOK'S DISTANCE ---

# Este método requiere un modelo de regresión. Crearemos uno simple.
model <- lm(feature2 ~ feature1 + feature3, data = train)

# Calcular la distancia de Cook
cooks_dist <- cooks.distance(model)

# El documento menciona un umbral de D > 1, pero D > 4/n es más común.
umbral_cook <- 4 / nrow(train)
outliers_cook_indices <- which(cooks_dist > umbral_cook)

cat("\n--- Distancia de Cook ---\n")
cat("Umbral de Distancia de Cook (4/n):", umbral_cook, "\n")
cat("Índices de Outliers detectados:", outliers_cook_indices, "\n")
print(train[outliers_cook_indices, ])

#----------------------------------------------------------------#
# 4. CREACIÓN DEL DATAFRAME FINAL CON OUTLIERS DETECTADOS
#----------------------------------------------------------------#

# Usaremos los resultados del Isolation Forest, el método recomendado.

# Añadimos las puntuaciones y una columna lógica para identificar outliers
train_with_outliers <- train %>%
  add_column(anomaly_score = predicciones_iso$anomaly_score) %>%
  mutate(is_outlier = anomaly_score >= umbral_iso)

cat("\n--- Dataframe Final con Outliers Detectados (sin eliminar) ---\n")
print(head(train_with_outliers))

cat("\n--- Filas identificadas como Outliers ---\n")
print(filter(train_with_outliers, is_outlier == TRUE))

