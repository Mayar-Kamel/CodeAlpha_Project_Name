#A/B testing
#Background
#we have results of A/B tests from two hotel booking websites . 
#First, we need to conduct a test analysis of the data; 
#second, we need to draw conclusions from the data which we obtained from the first step,
#and in the final step, we make recommendations or suggestions to the product or management teams.

#explanation of the data columns
#Variant A is from the control group which tells the existing features or products on a website.
#Variant B is from the experimental group to check the new version of a feature or 
#product to see if users like it or if it increases the conversions(bookings).
#Converted is based on the data set given, there are two categories defined by logical value. 
#It’s going to show true when the customer completes bookings 
#and it’s going to show false when the customer visits the sites but doesn't make a booking.

#Hypothesis
#Null Hypothesis: Both versions A and B have an equal probability of conversion 
#or driving customer booking. 
#In other words, there is no difference or no effect between A and B versions
#Alternative Hypothesis: Versions both A and B possess different probability of conversion or
#driving customer booking and there is a difference between A and B version. 
#Version B is better than version A in driving customer bookings. PExp_B! = Pcont_A.


# load the library as it  provides a set of tools for handling, manipulating, and visualizing
# data in a consistent and efficient manner. 
library(tidyverse) 


# Using read.csv base import function   

ABTest <- read.csv("//Users//mayar//Downloads//Website Results.csv",  
                   header = TRUE) 
# To see the structure of the data like number of rows and columns
str(ABTest)
# we see that there are 1451 rows and 4 columns

#Let's filter out conversions for variant_A which means that visitors made a booking
conversion_subset_A <- ABTest %>%  
    filter(variant == "A" & converted == "TRUE") 
conversion_subset_A

# Total Number of Conversions for variant_A 
conversions_A <- nrow(conversion_subset_A) 
conversions_A
# 20 customers from variant A made a booking

# Number of Visitors for variant_A 
visitors_A <- nrow(ABTest %>%  
                     filter(variant == "A")) 
visitors_A

# Conversion_rate_A by dividing the the conversions of A which is the number of visitors
# made abooking over the total number of visitors of variant A
conv_rate_A <- (conversions_A/visitors_A)   
print(conv_rate_A) 
#The probability is 0.0277

# Let's take a subset of conversions for variant_B 
conversion_subset_B <- ABTest %>%  
  filter(variant == "B" & converted == "TRUE")
conversion_subset_B
# Number of Conversions for variant_B 
conversions_B <- nrow(conversion_subset_B) 
conversions_B
# total numbers of visitors who made a booking from Variant B is 37

# Number of Visitors for variant_B 
visitors_B <- nrow(ABTest %>%  
                     filter(variant == "B")) 
visitors_B
# Conversion_rate_B 
conv_rate_B <- (conversions_B/visitors_B)   
print(conv_rate_B) 
# It is 0.0506
#conclusion:That conversion rate of variant B is higher than A which means that made more 
#bookings

#The relative uplift using conversion rates A & B. The uplift is a percentage of the increase
#It's commonly used to measure the relative improvement or change in conversion rates. 
uplift <- (conv_rate_B - conv_rate_A) / conv_rate_A * 100 
uplift 
#It is 82.72% which means that variant B is better than variant A by 82.72%

# Pooled sample proportion for variants A & B 
#This calculates the pooled sample proportion (p_pool) by adding the total number of conversions 
#for both variants (A and B) and dividing it by the total number of visitors for both
#variants. 
p_pool <- (conversions_A + conversions_B) / (visitors_A + 
                                               visitors_B) 
print(p_pool) 
#it is 0.03928325

# Let's compute Standard error for variants A & B (SE_pool) 
#the standard error for the pooled proportion (SE_pool) is calculated using the formula 
#for the standard error of a proportion. It takes into account the sample proportions 
#and sample sizes of both variants. 
SE_pool <- sqrt(p_pool * (1 - p_pool) * ((1 / visitors_A) +  
                                           (1 / visitors_B))) 
print(SE_pool)
# it is 0.01020014 

# Let's compute the margin of error for the pool 
#The margin of error (MOE) is computed by multiplying the standard error by 
#the critical z-value associated with a 95% confidence interval.
MOE <- SE_pool * qnorm(0.975) 
print(MOE) 
#it is 0.0199919 

# Point Estimate or Difference in proportion 
#This calculates the point estimate or the difference in proportions between variant B 
#and variant A (d_hat). It's the observed difference in conversion rates between the 
#two groups.
d_hat <- conv_rate_B - conv_rate_A
d_hat
#it is 0.02294568

#computing the z score to determine the p_value
#The z-score is often used in hypothesis testing to assess
#how many standard deviations an observed value is from the mean.
z_score <- d_hat / SE_pool 
print(z_score) 
#it is  2.249546


# Let's compute p_value using the z_score value 
p_value <- pnorm(q = -z_score,  
                 mean = 0,  
                 sd = 1) * 2 
print(p_value)
#it is 0.02447777 which is less than 0.05 which means reject null hypothesis
# and accept Ha which means that both  Versions  A and B possess different probability of conversion or
#driving customer booking and there is a difference between A and B version. 


#let’s visualize the results computed so far in a dataframe (table):

vis_result_pool <- data.frame( 
  metric = c( 
    'Estimated Difference', 
    'Relative Uplift(%)', 
    'pooled sample proportion', 
    'Standard Error of Difference', 
    'z_score', 
    'p-value', 
    'Margin of Error'), 
  value = c( 
    conv_rate_B - conv_rate_A, 
    uplift, 
    p_pool, 
    SE_pool, 
    z_score, 
    p_value, 
    MOE
  )) 
vis_result_pool

#Recommendations&conclusions
#Analysis Results: A/B Test for Variants A and B

#We conducted an A/B test comparing Variant A and Variant B based on the following metrics:
  
#Variant A: 20 conversions out of 721 hits, with a conversion rate of 2.77%.
#Variant B: 37 conversions out of 730 hits, with a conversion rate of 5.07%.
#Relative Uplift:
#The relative uplift in conversion rate for Variant B compared to Variant A is impressive,
#standing at 82.72%. Specifically, Variant B demonstrates a substantial improvement with a conversion rate
#of 5.07%, outperforming Variant A's 2.77%.

#Statistical Significance:
#The calculated p-value for this analysis is 0.02448, indicating strong statistical significance. 
#Given this result, we have robust evidence to reject the null hypothesis, 
#suggesting that the observed difference in conversion rates between the variants is not due to chance.

#Conclusion:
#With a high level of confidence in the statistical significance of our findings,
#we recommend accepting Variant B. The data strongly supports the notion that Variant B is 
#superior to Variant A, and we recommend moving forward with the launch of Variant B.

#Implementation Recommendation:
#Considering the compelling results, we propose rolling out Variant B to all users with confidence in 
#its improved performance.


