#########################################
##-----Setup file for the project-----##
########################################

# Load packages
packages <- c(
  "ggplot2",
  "tidyr",
  "dplyr",
  "psych",
  "inspectdf",
  "patchwork",
  "Metrics",
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

