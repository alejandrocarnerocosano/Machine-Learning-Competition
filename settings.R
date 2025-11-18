#########################################
##-----Setup file for the project-----##
########################################

# Load packages
packages <- c(
  "arules",
  "arulesViz",
  "caret",
  "dplyr",
  "FactoMineR",
  "fastDummies",
  "ggcorrplot",
  "ggplot2",
  "glue",
  "inspectdf",
  "lifecycle",
  "Metrics",
  "mice",
  "naniar",
  "patchwork",
  "PerformanceAnalytics",
  "psych",
  "Rlof",
  "solitude",
  "tidyr",
  "VIM"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Data Paths
path_raw <- "Data/raw"
path_intermediate <- "Data/intermediate"

# Theme
theme_set(theme_minimal())

