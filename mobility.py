# Mail id: kedharnath1992@gmail.com
# Kindly cite if you use the script ""



#############################################
# Import necessary modules
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import math
from math import log
from sklearn.metrics import mean_squared_error

#############################################
# Read the input data and select data for analysis
df = pd.read_excel("data.xlsx")
df.head()

pearcol = df.iloc[:, 0:5]    # In python the column numbering starts from zero
X = df.drop(df.columns[[5]], axis ='columns')
Y = df.Velocity
list(X.columns)

# FINDING activation energy (Q) & activation volume (Va) from data.xlsx


def act_ene(T1, T2, v1, v2):
  kB = 1.380649 * math.pow(10,-23)  # unit : J/K
  return -(kB * T1 * T2 * np.log(v1/v2)) / (T1-T2)    # Unit of Q : J

shear_tau = [0.5, 1, 1.5]
for j in range(0,len(shear_tau)):
  for i in range(0,len(X)):
    if df.W[i] == 5:
      if df.Slip_plane[i] == 123:
        if df.Applied_stress[i] == shear_tau[j]:
          if df.Temperature[i] == 1000:
            T1 = df.Temperature[i]
            v1 = df.Velocity[i] * 100     # convert to m/s
            #print(T1)
          if df.Temperature[i] == 2000:
            T2 = df.Temperature[i]
            v2 = df.Velocity[i] * 100
            #print(T2)
            Q = act_ene(T1, T2, v1, v2)
           # print(Q*(math.pow(10, 22)))   # This is for value to origin plotting software
# J to eV, then multiply with 6.242 * math.pow(10,18)

def act_vol(T, v1, v2, tau1, tau2):
  kB = 1.380649 * math.pow(10,-23)
  return (kB * T * np.log(v2/v1)) / (tau2 - tau1)

temp = [300, 1000, 2000]
for j in range(0,len(temp)):
  for i in range(0,len(X)):
    if df.W[i] == 10:
      if df.Slip_plane[i] == 110:
        if df.Temperature[i] == temp[j]:
          T = df.Temperature[i]
          if df.Applied_stress[i] == 1:
            tau1 = df.Applied_stress[i] * math.pow(10, 9)    # convert to Pa as 1 J = 1 Pa.m3
            v1 = df.Velocity[i] * 100     # convert to m/s
            #print(T1)
          if df.Applied_stress[i] == 1.5:
            tau2 = df.Applied_stress[i] * math.pow(10, 9)
            v2 = df.Velocity[i] * 100
            #print(T2)
            va = act_vol(T, v1, v2, tau1, tau2)
            #print(va)   # This is for value to origin plotting software
            va_b3 = (va* math.pow(10,27)) / (0.2863 * 0.2863 * 0.2863)
            print(va_b3)

#############################################
# Pearson correlation plot

cormat = pearcol.corr()
round(cormat,2)
fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
ax.figure.axes[-1].yaxis.label.set_size(20)

sns.set(font_scale=2)
plt.xticks(weight = 'bold', fontsize= 15)
plt.yticks(weight = 'bold', fontsize= 15)
plt.rcParams["axes.linewidth"] = 1.5
sns.heatmap(cormat, annot=True, linewidth=.5, ax = ax, fmt=".2f", cmap='coolwarm', cbar_kws={'shrink': 0.8}, annot_kws={'size': 15}, square=True);
for text in ax.texts:
    text.set_size(14)
    if text.get_text() >= '0.7':
        text.set_size(18)
        text.set_weight('bold')
        text.set_style('italic')
plt.savefig('pearson.png',dpi = 300, bbox_inches = "tight")


!pip install  "xgboost==2.0.1"
import xgboost as xgb
xgb.__version__  # version = 2.0.1

#!pip show scikit-learn
#!pip install scikit-learn --upgrade
#from sklearn import model_selection
#from model_selection import EarlyStopping

################################
#       XGBOOST   Optimization
################################

xgb.__version__  # version = 2.0.1

# Read the input data and select data for analysis
df = pd.read_excel("data.xlsx")
df.head()

pearcol = df.iloc[:, 0:6]    # In python the column numbering starts from zero
X = df.drop(df.columns[[4,5]], axis ='columns')
Y = df.Velocity
list(X.columns)


################################
#         XGBOOST
################################

# Read the input data and select data for analysis
df = pd.read_excel("data.xlsx")
df.head()

pearcol = df.iloc[:, 0:6]    # In python the column numbering starts from zero
X = df.drop(df.columns[[4,5]], axis ='columns')
Y = df.Velocity
list(X.columns)

# Separate training and testing data sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10)   # 25% data for testing
list(X_test)

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', use_rmm = 'True',
                          max_depth = 4,
                          #early_stopping_rounds = 18,
                          #gamma = 4,
                          #colsample_bytree = 0.757576,
                          subsample = 0.242424,
                          learning_rate =  0.383838,
                          #max_delta_step = 2,
                          #random_state = 1,
                          #n_estimators = 14,
                          #alpha = 6,
                          #seed = 85
                          )

xg_reg.fit(X_train,Y_train)
#xg_reg.fit(X_train,Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)])

#es = EarlyStopping(monitor='val_rmse', patience=18)
#xg_reg.fit(X_train, Y_train, early_stopping=es)


preds = xg_reg.predict(X_test)     # predict y data using X_test
# Low RMSE is desired
rmse_test = np.sqrt(mean_squared_error(Y_test, preds))  # calc RMSE bet y_test & y_predict
print("RMSE testing: %f" % (rmse_test))

############# Plotting

xgb.plot_importance(xg_reg)
#plt.savefig('xgb feature imp.png',dpi = 300)
plt.rcParams['figure.figsize'] = [5,5]

pred_ytrain = xg_reg.predict(X_train)
#train = np.dstack((Y_train, pred_ytrain))
pred_ytest = xg_reg.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(Y_train, pred_ytrain))
print("RMSE training: %f" % (rmse_train))

# Pearson linear correlation coefficent
corr_test, _ = pearsonr(Y_test, preds)
print('Pearsons correlation test: %.3f' % corr_test)
corr_train, _ = pearsonr(Y_train, pred_ytrain)
print('Pearsons correlation train: %.3f' % corr_train)

Test_accuracy = xg_reg.score(X_test, Y_test)
print('Test accuracy: %.3f' % Test_accuracy)
Train_accuracy = xg_reg.score(X_train, Y_train)
print('Train accuracy: %.3f' % Train_accuracy)


############# Plotting
plt.show()
plt.scatter(Y_train, pred_ytrain, c='b', label='Training 75% data', s = 70, alpha=0.5)
plt.scatter(Y_test, pred_ytest, c='r', label='Testing 25% data', s = 70, alpha=0.5)
plt.plot([7, 18], [7, 18], color = 'black', linewidth = 3, linestyle='dashed')
plt.xlim(7, 18)
plt.ylim(7, 18)

#plt.legend(loc="upper left", fontsize = 6, prop={'weight': 'bold'})
plt.xticks(weight = 'bold', fontsize= 10)
plt.yticks(weight = 'bold', fontsize= 10)
plt.rcParams["axes.linewidth"] = 1.5
plt.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)

plt.xlabel('MD', fontsize= 20, fontweight='bold')
plt.ylabel('XGBoost', fontsize= 20, fontweight='bold')
plt.savefig('xgb.png',dpi = 300, bbox_inches = "tight")

!pip install shap
import shap as shap
#import xgboost as xgb
#pip install xgboost==2.0.1
#shap.__version__ # version = 0.43.0

################################
# SHAP analysis
################################


list(X.columns)
# X = X1.drop(X1.columns[[2,3]], axis ='columns')  # SHAP needs same features as XGBoost
shap.initjs()
explainer = shap.Explainer(xg_reg, X) # (xg_reg, X_test), (xg_reg, X_train), & (xg_reg) shows DIFFERENCE
shap_values = explainer(X)   # if X_test is changed to X_train then shows DIFFERENCE; but mostly X_test is used https://www.kaggle.com/code/dansbecker/shap-values
feature_names = [ a + ": " + str(b) for a,b in zip(X.columns, np.abs(shap_values.values).mean(0))]
shap.summary_plot(shap_values, plot_type='violin', feature_names = feature_names, color_bar = False, show = False, alpha=0.5)


cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=15)
cbar.ax.tick_params(direction='out', length=6, width=2,  grid_alpha=0.5)

plt.xticks(weight = 'bold', fontsize= 13)
plt.yticks(weight = 'bold', fontsize= 13)
plt.rcParams["axes.linewidth"] = 1.5
plt.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
plt.xlabel('SHAP value', fontsize= 15, fontweight='bold')

plt.savefig("shap violin.png", dpi = 300, bbox_inches = "tight")
plt.close()

shap.summary_plot(shap_values, plot_type='violin', feature_names=feature_names)


shap.plots.scatter(shap_values[:,"W"], color=shap_values[:,"W"], dot_size=80, cmap='rainbow', alpha=0.5)
plt.xticks(weight = 'bold', fontsize= 13)
plt.yticks(weight = 'bold', fontsize= 13)
plt.rcParams["axes.linewidth"] = 1.5
plt.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
plt.savefig("shap W.png", dpi = 300, bbox_inches = "tight")
plt.close()

shap.plots.scatter(shap_values[:,"Slip_plane"], color=shap_values[:,"Slip_plane"], dot_size=80, cmap='rainbow', alpha=0.5)
plt.xticks(weight = 'bold', fontsize= 13)
plt.yticks(weight = 'bold', fontsize= 13)
plt.rcParams["axes.linewidth"] = 1.5
plt.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
plt.savefig("shap slip.png", dpi = 300, bbox_inches = "tight")
plt.close()

shap.plots.scatter(shap_values[:,"Temperature"], color=shap_values[:,"Temperature"], dot_size=80, cmap='rainbow', alpha=0.5)
plt.xticks(weight = 'bold', fontsize= 13)
plt.yticks(weight = 'bold', fontsize= 13)
plt.rcParams["axes.linewidth"] = 1.5
plt.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
plt.savefig("shap temp.png", dpi = 300, bbox_inches = "tight")
plt.close()

shap.plots.scatter(shap_values[:,"Applied_stress"], color=shap_values[:,"Applied_stress"], dot_size=80, cmap='rainbow', alpha=0.5)
plt.xticks(weight = 'bold', fontsize= 13)
plt.yticks(weight = 'bold', fontsize= 13)
plt.rcParams["axes.linewidth"] = 1.5
plt.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)
plt.savefig("shap stress.png", dpi = 300, bbox_inches = "tight")
plt.close()

