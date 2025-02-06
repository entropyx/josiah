install.packages("remotes") 
remotes::install_github(
  repo = "facebookexperimental/siMMMulator"
)

library(dplyr)
library(siMMMulator)


my_variables <- step_0_define_basic_parameters(years = 4,
                                               channels_impressions = c("Facebook", "TV","Youtube","OOH","Radio","ProgammaticRT","PMAX"),
                                               channels_clicks = c("Search"),
                                               frequency_of_campaigns = 12,
                                               true_cvr = c(0.002, 0.002, 0.002,0.003,0.003,0.001,0.003,0.04),
                                               revenue_per_conv = 15, 
                                               start_date = "2020/1/1"
)
df_baseline <- step_1_create_baseline(
  my_variables = my_variables,
  base_p = 500000,
  trend_p = 1.5,
  temp_var = 4,
  temp_coef_mean = 50000,
  temp_coef_sd = 5000,
  error_std = 100000)

optional_step_1.5_plot_baseline_sales(df_baseline = df_baseline)

df_ads_step2 <- step_2_ads_spend(
  my_variables = my_variables,
  campaign_spend_mean = 229000,
  campaign_spend_std = 50000,
  max_min_proportion_on_each_channel <- c(0.2,0.3,
                                          0.05,0.1,
                                          0.02,0.04,
                                          0.1,0.15,
                                          0.15,0.2,
                                          0.03,0.05,
                                          0.05,0.07,
                                          0.02,0.03)
)

optional_step_2.5_plot_ad_spend(df_ads_step2 = df_ads_step2)



df_ads_step3 <- step_3_generate_media(
  my_variables = my_variables,
  df_ads_step2 = df_ads_step2,
  true_cpm = c(2, 25, 10, 15, 10, 8, 7, NA),
  true_cpc = c(NA, NA, NA, NA, NA,NA, NA, 0.5),
  mean_noisy_cpm_cpc = c(1, 0.05, 0.01, 0.02, 0.03, 0.02, 0.02, 0.02),
  std_noisy_cpm_cpc = c(0.01, 0.15, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02)
)

df_ads_step4 <- step_4_generate_cvr(
  my_variables = my_variables,
  df_ads_step3 = df_ads_step3,
  mean_noisy_cvr = c(0.0001, 0.0001, 0.0002, 0.0001, 0.0002, 0.0001, 0.0001, 0.002), 
  std_noisy_cvr = c(0.001, 0.002, 0.003, 0.003, 0.004, 0.002, 0.002, 0.001)
)

df_ads_step5a_before_mmm <- step_5a_pivot_to_mmm_format(
  my_variables = my_variables,
  df_ads_step4 = df_ads_step4
)

# c <- c("Facebook", "TV","Youtube","OOH","Radio","ProgammaticRT","PMAX","Search")

df_ads_step5b <- step_5b_decay(
  my_variables = my_variables,
  df_ads_step5a_before_mmm = df_ads_step5a_before_mmm,
  true_lambda_decay = c(0.1, 0.5, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1)
)


df_ads_step5c <- step_5c_diminishing_returns(
  my_variables = my_variables,
  df_ads_step5b = df_ads_step5b,
  alpha_saturation = c(2.2, 3.0, 2.4, 3.5, 3.2, 2.5, 2.3, 1.5),
  gamma_saturation = c(0.5, 0.3, 0.4, 0.25, 0.3, 0.4, 0.45, 0.7)
)

df_ads_step6 <- step_6_calculating_conversions(
  my_variables = my_variables,
  df_ads_step5c = df_ads_step5c
)

df_ads_step7 <- step_7_expanded_df(
  my_variables = my_variables,
  df_ads_step6 = df_ads_step6,
  df_baseline = df_baseline
)

step_8_calculate_roi(
  my_variables = my_variables,
  df_ads_step7 = df_ads_step7
)

list_of_df_final <- step_9_final_df(
  my_variables = my_variables,
  df_ads_step7 = df_ads_step7
)

daily_df <- list_of_df_final[[1]]
optional_step_9.5_plot_final_df(df_final = list_of_df_final[[1]])

weekly_df <- list_of_df_final[[2]]
optional_step_9.5_plot_final_df(df_final = list_of_df_final[[2]])

# Add Promo Effects

weekly_df$DATE <- as.Date(weekly_df$DATE)

# Extract the week number
weekly_df$week <- as.numeric(format(weekly_df$DATE, "%U")) + 1  # Week number from DATE

# Define promo weeks
promo1_weeks <- c(3, 25, 45)
promo2_weeks <- c(8, 35, 36, 50)

# Apply promotions
weekly_df <- weekly_df %>%
  mutate(
    promo1 = ifelse(week %in% promo1_weeks, 1, 0),
    promo2 = ifelse(week %in% promo2_weeks, 1, 0),
    total_revenue = total_revenue * ifelse(promo1 == 1, runif(nrow(weekly_df), 1.08, 1.12), 1),
    total_revenue = total_revenue * ifelse(promo2 == 1, runif(nrow(weekly_df), 1.04, 1.06), 1)
  )

# Display first rows
head(weekly_df)
plot(weekly_df$total_revenue, type="l")


write.csv(weekly_df, "~/Development/simmmulator/weekly.csv", row.names = TRUE)

