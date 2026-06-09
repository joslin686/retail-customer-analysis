# 1. Load libraries
library(tidyverse)
library(lubridate)
library(cluster)
library(factoextra)

# 2. Load dataset
# Ensure the file is in your working directory
df <- read.csv("Online_Retail_Clean (2).csv")

# 3. Check data
print(str(df))
print(head(df))
print(summary(df))

# 1. Convert InvoiceDate to proper POSIXct format
df$InvoiceDate <- as.POSIXct(df$InvoiceDate, format="%Y-%m-%d %H:%M:%S")

# 2. Define Reference Date (1 day after the latest transaction)
# This will be our "current" date for the RFM calculation
ref_date <- max(df$InvoiceDate) + days(1)

# Check the conversion
print(paste("Reference Date:", ref_date))
print(str(df$InvoiceDate))

# 3. Create RFM Dataframe
rfm <- df %>%
  group_by(CustomerID) %>%
  summarise(
    Recency = as.numeric(difftime(ref_date, max(InvoiceDate), units = "days")),
    Frequency = n_distinct(InvoiceNo),
    Monetary = sum(TotalPrice)
  )

# 4. Check the results
print(head(rfm))
print(summary(rfm))
# 1. Create RFM Dataframe
rfm <- df %>%
  group_by(CustomerID) %>%
  summarise(
    Recency = as.numeric(difftime(ref_date, max(InvoiceDate), units = "days")),
    Frequency = n_distinct(InvoiceNo),
    Monetary = sum(TotalPrice)
  )

# 2. Check the results
print(head(rfm))
print(summary(rfm))

# 1. Reshape data for plotting
rfm_long <- rfm %>%
  select(Recency, Frequency, Monetary) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

# 2. Create histograms
ggplot(rfm_long, aes(x = Value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of RFM Features",
       subtitle = "Notice the heavy right-skew (long tail) in Frequency and Monetary",
       x = "Value", y = "Count")

# 1. Log Transformation to handle skewness
# We select columns 2, 3, 4 (Recency, Frequency, Monetary)
rfm_log <- rfm %>%
  mutate(
    Recency = log1p(Recency),
    Frequency = log1p(Frequency),
    Monetary = log1p(Monetary)
  )

# 2. Scaling
# Clustering is sensitive to scale; we need all variables to have the same mean/variance
rfm_scaled <- scale(rfm_log[, 2:4])

# Check the results
summary(rfm_scaled)

# 1. Run PCA
pca_result <- prcomp(rfm_scaled)

# 2. Scree Plot (to visualize variance explained)
# fviz_eig is from the factoextra library we loaded earlier
p_scree <- fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 100)) +
  labs(title = "Scree Plot", x = "Principal Components", y = "% of Variance Explained")

# Print the scree plot
print(p_scree)

# 3. Summary of PCA
print(summary(pca_result))
# 1. Perform K-Means clustering with k = 2
set.seed(42) 
km_result_2 <- kmeans(rfm_scaled, centers = 2, nstart = 25)

# 2. Add to dataframe
rfm$Cluster_2 <- as.factor(km_result_2$cluster)

# 3. Cluster Averages for k = 2
summary_k2 <- rfm %>%
  group_by(Cluster_2) %>%
  summarise(
    Avg_Recency = mean(Recency),
    Avg_Frequency = mean(Frequency),
    Avg_Monetary = mean(Monetary),
    Count = n()
  )

print("Cluster Summary for k = 2:")
print(summary_k2)

# 4. Visualize k = 2
p_clusters_2 <- fviz_cluster(km_result_2, data = rfm_scaled,
                             geom = "point",
                             ellipse.type = "convex",
                             ggtheme = theme_minimal()) +
  labs(title = "Customer Segments (k=2)")

print(p_clusters_2)

# 1. Perform K-Means clustering with k = 3
set.seed(42) 
km_result_3 <- kmeans(rfm_scaled, centers = 3, nstart = 25)

# 2. Add to dataframe
rfm$Cluster_3 <- as.factor(km_result_3$cluster)

# 3. Cluster Averages for k = 3
summary_k3 <- rfm %>%
  group_by(Cluster_3) %>%
  summarise(
    Avg_Recency = mean(Recency),
    Avg_Frequency = mean(Frequency),
    Avg_Monetary = mean(Monetary),
    Count = n()
  )

print("Cluster Summary for k = 3:")
print(summary_k3)

# 4. Visualization for k = 3
p_clusters_3 <- fviz_cluster(km_result_3, data = rfm_scaled,
                             geom = "point",
                             ellipse.type = "convex",
                             ggtheme = theme_minimal()) +
  labs(title = "Customer Segments (k=3)")

print(p_clusters_3)

