# --- Step 1: Data Preparation for Time Series ---

# Load required libraries
library(tseries)

# 1.1 Load the clean dataset
data <- read.csv("Online_Retail_Clean.csv")

# 1.2 Convert InvoiceDate to Date format (ignoring time)
data$Date <- as.Date(data$InvoiceDate)

# 1.3 Aggregate TotalPrice by Date to get Daily Sales
daily_sales <- aggregate(TotalPrice ~ Date, data, sum)

# 1.4 Convert to a Time Series (ts) object
# Since we have daily data, we set frequency = 365
start_year <- as.numeric(format(min(daily_sales$Date), "%Y"))
start_day <- as.numeric(format(min(daily_sales$Date), "%j"))

sales_ts <- ts(daily_sales$TotalPrice, 
               start = c(start_year, start_day), 
               frequency = 365)

# 1.5 Initial Exploratory Plotting (EDA)
print(start(sales_ts))
print(end(sales_ts))
summary(sales_ts)

# Plot the time series
plot(sales_ts, 
     main = "Daily Total Sales (Online Retail)", 
     xlab = "Time", 
     ylab = "Total Sales", 
     col = "blue")

# Add a trend line (Simple Linear Regression)
abline(reg = lm(sales_ts ~ time(sales_ts)), col = "red")

# --- Step 2: Transformation and Smoothing ---

# 2.1 Log Transformation
sales_ts_log <- log(sales_ts)

# Plot the log-transformed series
plot.ts(sales_ts_log, 
        main = "Log-Transformed Daily Sales", 
        col = "darkgreen", 
        ylab = "Log(Sales)")

# 2.2 Smoothing using Simple Moving Average (SMA)
library(TTR)

# n=7 represents a weekly smoothing (since our data is daily)
sales_sma7 <- SMA(sales_ts, n = 7)

# n=30 represents a monthly smoothing
sales_sma30 <- SMA(sales_ts, n = 30)

# Plot the smoothed versions to see the trend clearly
plot.ts(sales_sma7, main = "7-Day Moving Average (Weekly Trend)", col = "red")
plot.ts(sales_sma30, main = "30-Day Moving Average (Monthly Trend)", col = "blue")

# --- Step 3: Seasonality and Decomposition ---

# 3.1 Add DayOfWeek back to the aggregated daily_sales dataframe
daily_sales$DayOfWeek <- weekdays(daily_sales$Date)

# Reorder days so they appear in order (Monday to Sunday) in the plot
daily_sales$DayOfWeek <- factor(daily_sales$DayOfWeek, 
                                levels=c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

# 3.2 Re-create the Time Series with frequency = 7 (Weekly Seasonality)
sales_ts_7 <- ts(daily_sales$TotalPrice, frequency = 7)
sales_ts_log_7 <- log(sales_ts_7)

# 3.3 Boxplot
boxplot(TotalPrice ~ DayOfWeek, data = daily_sales, 
        main = "Sales Distribution by Day of the Week",
        col = "orange",
        xlab = "Day", ylab = "Total Sales")

# 3.4 Decomposing the Time Series
sales_components <- decompose(sales_ts_log_7)

# Plot the individual components: Observed, Trend, Seasonal, and Random
plot(sales_components, col = "purple")

# 3.5 Seasonally Adjusted Data
sales_adjusted <- sales_ts_log_7 - sales_components$seasonal
plot(sales_adjusted, main = "Seasonally Adjusted Sales (Weekly)", col = "brown")

# --- Step 4: Stationarity and ACF/PACF ---

# 4.1 Augmented Dickey-Fuller (ADF) Test
# This tests if the data is stationary. 
# We want a p-value LESS than 0.05.
adf.test(sales_ts_log_7)

# 4.2 Differencing
sales_diff <- diff(sales_ts_log_7)

# Check ADF again on differenced data
adf.test(sales_diff)

# 4.3 ACF and PACF Plots (Task 5)
# These help identify the ARIMA parameters (p, d, q)
par(mfrow = c(1, 2)) # Show two plots side-by-side
acf(sales_diff, main = "ACF Plot")
pacf(sales_diff, main = "PACF Plot")
par(mfrow = c(1, 1)) # Reset plot layout

# --- Step 5: Fitting ARIMA and Forecasting ---
library(forecast)

# 5.1 Fit the ARIMA model
# 'period = 7' tells the model to respect the weekly pattern we found
fit <- arima(sales_ts_log_7, 
             order = c(5, 1, 0), 
             seasonal = list(order = c(1, 1, 0), period = 7))

# 5.2 Make a prediction for the next 30 days
sales_forecast <- forecast(fit, h = 30)

# 5.3 Plot the forecast
plot(sales_forecast, 
     main = "30-Day Sales Forecast (Weekly Pattern)", 
     xlab = "Time (Weeks)", 
     ylab = "Log(Sales)", 
     col = "blue", 
     fcol = "red")

# 5.4 Accuracy Metrics
accuracy(fit)

# --- Comparison: Searching for the 'Best' Model ---
library(forecast)

# 1. Let R find the "mathematically best" ARIMA model automatically
auto_fit <- auto.arima(sales_ts_log_7, seasonal = TRUE)
print(auto_fit)

# 2. ETS (Exponential Smoothing) model
ets_fit <- ets(sales_ts_log_7)
print(ets_fit)

# 3. Compare Accuracy
accuracy(fit)
accuracy(auto_fit)
accuracy(ets_fit)

# --- Final Optimized Forecast (Using Auto-ARIMA) ---

# 1. Forecast 30 days ahead using the best model
final_auto_forecast <- forecast(auto_fit, h = 30)

# 2. Convert back to original scale (Dollars)
final_auto_forecast$mean <- exp(final_auto_forecast$mean)
final_auto_forecast$lower <- exp(final_auto_forecast$lower)
final_auto_forecast$upper <- exp(final_auto_forecast$upper)
final_auto_forecast$x <- exp(final_auto_forecast$x)

# 3. Final Plot
plot(final_auto_forecast, 
     main = "Optimized 30-Day Sales Forecast (ARIMA 5,1,3)", 
     ylab = "Total Sales ($)", 
     xlab = "Time (Weeks)", 
     col = "black", fcol = "red", shadecols = "oldstyle")

