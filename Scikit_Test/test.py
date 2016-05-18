'''
Following Lab 4 of Harvard's CS109 class
'''

import numpy as np
import pandas as pd # R-Style Dataframes
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_lin

# Linear Regression lib
import statsmodels.api as sm
import statsmodels.formula.api as sm_form

import seaborn as sns # Pretty Plots
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams

from sklearn.datasets import load_boston

# Loads sample dataset into a sklearn data structure
boston = load_boston()
'''
    boston.data             = feature data as array
    boston.feature_names    = feature names
    boston.DESCR            = Text description of data
    boston.target           = Output variable training data (Price in this case)
'''

print("Boston Features:")
print boston.feature_names
print("Boston Description:")
print boston.DESCR

# Put the data into a pandas dataframe for easy statistic gathering
bos_df = pd.DataFrame(boston.data)

# Add in column names to df
bos_df.columns = boston.feature_names

# add in the response variable
bos_df["PRICE"] = boston.target

stats = bos_df.describe()
crim_mean = stats.get("CRIM").get("count") 

plt.scatter(bos_df.CRIM,bos_df.PRICE) # This works as well as the <get> method
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")
plt.show()

# Use seaborn to do a plot and automatic lin regression
sns.regplot(y="PRICE", x="RM", data=bos_df, fit_reg = True)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")
plt.show()

''' Do a linear regressionusing statsmodels.api '''

R_formula_out = "PRICE ~ ";
R_formula_in = "+".join(boston.feature_names)
R_formula = "".join([R_formula_out,R_formula_in])

# R_formula = "PRICE ~ CRIM+ZN+INDUS+..."
# "Y ~ X1 + X2" means predict Y from features X1 and X2, assuming Y,X1,X2 are all names in a datafram
sm_lin_model = sm_form.ols(R_formula,bos_df).fit()
print sm_lin_model.summary()

predictions = sm_lin_model.fittedvalues

plt.scatter(bos_df['PRICE'], predictions)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()


''' Do the same with Scikit Learn with all features'''
X = bos_df.drop('PRICE',axis=1) # Create the feature dataframe without the price
sk_lin_model = sk_lin.LinearRegression() # Create new Lin Regression Model
sk_lin_model.fit(X,bos_df.get("PRICE"))
print 'Estimated intercept coefficient using Scikit:', sk_lin_model.intercept_
print 'Estimated number of coefficients using Scikit:', len(sk_lin_model.coef_)
