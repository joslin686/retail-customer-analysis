# ── Step 1: Enhanced Feature Engineering ───────────────
library(dplyr)

reference_date <- max(df$InvoiceDate, na.rm = TRUE)

customer_features <- df %>%
  group_by(CustomerID) %>%
  summarise(
    # RFM for Churn Logic ONLY
    DaysSinceLastPurchase = as.numeric(difftime(reference_date, max(InvoiceDate), units = "days")),
    PurchaseFrequency     = n_distinct(InvoiceNo),
    TotalSpent            = sum(TotalPrice),
    
    # --- MODEL INPUTS (The Clues) ---
    
    # 1. Product Diversity: How many unique products do they buy?
    # High diversity often correlates with higher brand loyalty.
    ProductDiversity      = n_distinct(StockCode),
    
    # 2. Average Basket Value: Average spend per invoice
    # Distinguishes big-ticket shoppers from small/frequent buyers.
    AvgBasketValue        = sum(TotalPrice) / n_distinct(InvoiceNo),
    
    # 3. Tenure: How long they've been with the store
    DaysSinceFirstPurchase = as.numeric(difftime(reference_date, min(InvoiceDate), units = "days")),
    
    # 4. Volume and Price Sensitivity
    AvgQuantity           = mean(Quantity),
    AvgUnitPrice          = mean(UnitPrice)
  ) %>%
  
  # Apply your Churn Logic
  mutate(
    IsChurned = as.factor(ifelse(
      DaysSinceLastPurchase > 90 | (PurchaseFrequency <= 3 & TotalSpent <= 100),
      "Yes", "No"
    ))
  )
cat("Total customers:", nrow(customer_features), "\n")
cat("Churned (Yes):", sum(customer_features$IsChurned == "Yes"), "\n")
cat("Not Churned (No):", sum(customer_features$IsChurned == "No"), "\n")
cat("Churn rate:", round(mean(customer_features$IsChurned == "Yes") * 100, 1), "%\n")
head(customer_features)

# ── Update Model Data Selection ────────────────────────
# Now we select the new features for training
# ── Step 2: Prepare data for modelling ────────────────────────────────────────
library(caret)
model_data <- customer_features %>%
  select(ProductDiversity, AvgBasketValue, DaysSinceFirstPurchase, 
         AvgQuantity, AvgUnitPrice, IsChurned)




# Select ONLY the 3 model input features + IsChurned
# RFM columns are excluded — they were only used to define IsChurned


cat("Dataset dimensions:", nrow(model_data), "rows,", ncol(model_data), "columns\n")
head(model_data)

# ── Train/test split (70/30) ──────────────────────────────────────────────────
set.seed(42)
ind <- sample(1:nrow(model_data), size = round(0.7 * nrow(model_data)))

train_data <- model_data[ind, ]
test_data  <- model_data[-ind, ]

cat("\nTraining set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

cat("\nClass distribution in training set:\n")
print(prop.table(table(train_data$IsChurned)) * 100)

cat("\nClass distribution in test set:\n")
print(prop.table(table(test_data$IsChurned)) * 100)


# ── Install & load libraries ───────────────────────────────────────────────────
if (!require(rpart)) install.packages("rpart")
if (!require(rpart.plot)) install.packages("rpart.plot")
if (!require(caret)) install.packages("caret")
library(rpart)
library(rpart.plot)
library(caret)


# ── Step 3.1: Train the Decision Tree ─────────────────────────────────────────

# rpart() builds the tree.
# IsChurned ~ . means "predict IsChurned using all other columns"
# method = "class" because IsChurned is categorical (Yes/No)
tree_model <- rpart(IsChurned ~ ., data = train_data, method = "class")


# ── Step 3.2: Examine the model ───────────────────────────────────────────────

# Text description of the tree — shows split rules, sample counts and probabilities
print(tree_model)

# CP table — shows how error changes with tree size
printcp(tree_model)
plotcp(tree_model)


# ── Step 3.3: Visualise the Decision Tree ─────────────────────────────────────

# Basic plot
plot(tree_model, uniform = TRUE, main = "Classification Tree for IsChurned")
text(tree_model, use.n = TRUE, all = TRUE, cex = 0.8)

# Nicer plot
rpart.plot(tree_model, type = 4, extra = 104, main = "Decision Tree - Customer Churn")


# ── Step 3.4: Make predictions ────────────────────────────────────────────────

# Predict Yes/No labels on the test set
tree_pred <- predict(tree_model, test_data, type = "class")


# ── Step 3.5: Evaluate the model ──────────────────────────────────────────────

# Basic confusion matrix table
cat("\n── Confusion Matrix ──\n")
print(table(tree_pred, test_data$IsChurned))

# Detailed evaluation — accuracy, precision, recall, F1
cat("\n── Detailed Evaluation ──\n")
conf_matrix_dt <- confusionMatrix(tree_pred, test_data$IsChurned, positive = "Yes")
print(conf_matrix_dt)


# ── Step 3.6: Prune the tree ──────────────────────────────────────────────────

# Pruning removes branches that overfit the training data.
# Picks the cp value with the lowest cross-validation error.
pruned_model <- prune(
  tree_model,
  cp = tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
)

# Visualise pruned tree
rpart.plot(pruned_model, type = 4, extra = 104, main = "Pruned Decision Tree - Customer Churn")

# Evaluate pruned model
pruned_pred <- predict(pruned_model, test_data, type = "class")

cat("\n── Pruned Tree Confusion Matrix ──\n")
print(table(pruned_pred, test_data$IsChurned))

cat("\n── Pruned Tree Detailed Evaluation ──\n")
conf_matrix_pruned <- confusionMatrix(pruned_pred, test_data$IsChurned, positive = "Yes")
print(conf_matrix_pruned)

# Export customer features to CSV for use in Python/Colab
write.csv(customer_features, "customer_features2.csv", row.names = FALSE)
cat("Saved to customer_features.csv\n")
print(tree_model$variable.importance)
