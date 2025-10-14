#2. Bivariant analysis before imputation

#Numerical vs. numerical
datos<-read.csv("train.csv")

sapply(datos, class)

datos$song_popularity<-as.numeric(datos$song_popularity)
datos$song_duration_ms<-as.numeric(datos$song_duration_ms)

datos$time_signature<-as.factor(datos$time_signature)
datos$key <-as.factor(datos$key)
datos$audio_mode  <-as.factor(datos$audio_mode)
datos$ID<-as.factor(datos$ID)



clases <- sapply(datos, class)
varNum <- names(clases)[which(clases %in% c("numeric", "integer"))]
varCat <- names(clases)[which(clases %in% c("character", "factor"))]

cor(datos[, varNum])
library(PerformanceAnalytics)
chart.Correlation(as.matrix(datos[, varNum]),histogram = TRUE,pch=12)

library(ggcorrplot)
corr <- round(cor(datos[, varNum]), 1)
ggcorrplot(corr, lab = T)

#No ha resultados


#2. Bivariant analysis after imputation
datos<-read.csv("imputedn.csv")

sapply(datos, class)

datos$song_popularity<-as.numeric(datos$song_popularity)
datos$song_duration_ms<-as.numeric(datos$song_duration_ms)

datos$time_signature<-as.factor(datos$time_signature)
datos$key <-as.factor(datos$key)
datos$audio_mode  <-as.factor(datos$audio_mode)
datos$ID<-as.factor(datos$ID)



clases <- sapply(datos, class)
varNum <- names(clases)[which(clases %in% c("numeric", "integer"))]
varCat <- names(clases)[which(clases %in% c("character", "factor"))]

cor(datos[, varNum])
library(PerformanceAnalytics)
chart.Correlation(as.matrix(datos[, varNum]),histogram = TRUE,pch=12)

library(ggcorrplot)
corr <- round(cor(datos[, varNum]), 1)
ggcorrplot(corr, lab = T)

#Ninguna de las variables muestra una correlación lineal significativa con song_popularity.

#Posible multicolinealidad: 
#energy y loudness tienen una fuerte correlación positiva (0.7). 
#Esto tiene mucho sentido, ya que las canciones con más energía suelen tener un volumen más alto.
#acousticness y energy tienen una fuerte correlación negativa (-0.7). 
#También es lógico: las canciones acústicas suelen ser menos enérgicas. Lo mismo ocurre entre acousticness y loudness (-0.6).
