# ── Install & load packages ───────────────────────────────────────────────────
if (!require(readxl))  install.packages("readxl")
if (!require(dplyr))   install.packages("dplyr")
if (!require(lubridate)) install.packages("lubridate")

library(readxl)
library(dplyr)
library(lubridate)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df <- read_excel("Online Retail.xlsx")
cat("Original rows:", nrow(df), "\n")

# ── 2. Remove missing CustomerID ─────────────────────────────────────────────
df <- df %>% filter(!is.na(CustomerID))
cat("After removing missing CustomerID:", nrow(df), "\n")

# ── 3. Remove cancelled orders (InvoiceNo starts with "C") ───────────────────
df <- df %>% filter(!grepl("^C", InvoiceNo))
cat("After removing cancellations:", nrow(df), "\n")

# ── 4. Remove negative or zero Quantity ──────────────────────────────────────
df <- df %>% filter(Quantity > 0)
cat("After removing bad Quantity:", nrow(df), "\n")

# ── 5. Remove zero or negative UnitPrice ─────────────────────────────────────
df <- df %>% filter(UnitPrice > 0)
cat("After removing bad UnitPrice:", nrow(df), "\n")

# ── 6. Remove missing Description ────────────────────────────────────────────
df <- df %>% filter(!is.na(Description) & Description != "")
cat("After removing missing Description:", nrow(df), "\n")

# ── 7. Remove duplicates ──────────────────────────────────────────────────────
df <- df %>% distinct()
cat("After removing duplicates:", nrow(df), "\n")

# ── 8. Create TotalPrice column ───────────────────────────────────────────────
df <- df %>% mutate(TotalPrice = Quantity * UnitPrice)

# ── 9. Fix date column & extract time features ────────────────────────────────
df <- df %>%
  mutate(
    InvoiceDate = as.POSIXct(InvoiceDate),
    Year        = year(InvoiceDate),
    Month       = month(InvoiceDate),
    DayOfWeek   = wday(InvoiceDate, label = TRUE)
  )

# ── 10. Final check ───────────────────────────────────────────────────────────
cat("\n── Final dataset summary ──\n")
glimpse(df)

# ── 11. Save cleaned dataset ──────────────────────────────────────────────────
write.csv(df, "Online_Retail_Clean.csv", row.names = FALSE)
cat("\nSaved to Online_Retail_Clean.csv\n")

#  ── 12. Visualizations──────────────────────────────────────────────────
library(ggplot2)
library(dplyr)
library(readxl)

df_raw <- read_excel("Online Retail.xlsx")

# ── Missing values per column ──────────────────────────────────────────────
missing_df <- data.frame(
  Column  = names(df_raw),
  Missing = colSums(is.na(df_raw)) / nrow(df_raw) * 100
)

ggplot(missing_df, aes(x = reorder(Column, -Missing), y = Missing)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Missing Values per Column", x = "Column", y = "% Missing") +
  theme_minimal()

# ──  Orders by month ────────────────────────────────────────────────────────
ggplot(df_clean, aes(x = factor(Month))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Number of Orders by Month", x = "Month", y = "Count") +
  theme_minimal()

# ── Orders by day of week ──────────────────────────────────────────────────
ggplot(df_clean, aes(x = DayOfWeek)) +
  geom_bar(fill = "coral") +
  labs(title = "Number of Orders by Day of Week", x = "Day", y = "Count") +
  theme_minimal()
