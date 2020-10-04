# Flipr-ML-Gryf1ndor
## Part 1:
### a) Data cleaning and preprocessing:

        1) Handling missing values:
            For columns who had less number of missing values, we filled them by their mean.
            For columns who had more number of missing values, we used groupby() and .mean() to fill values according to their group.

            Grouped by state- Foreign Visitors, Avg temp
            Grouped by Type – Toilets Avl, no. of hosps

        2) We plotted the correlation matrix and dropped the columns who had very poor relations with the target variable.
        3) Plotted Boxplot and handled the outliers
        4) Normalized the features
### b) Model making
We have used the **XgBoost** model for predicting the Covid cases here because boosting is an ensemble technique where new models are added to correct the errors made by existing models. It works well when there is not so much noice in the dataset and also has a lot of parameters to tune to help fit the model.
The parameters we applied are:<br/>
`model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                                     learning_rate=0.05, max_depth=5, 
                                     min_child_weight=1.7817, n_estimators=2200,
                                     reg_alpha=0.4640, reg_lambda=0.8571,
                                     subsample=0.5213, silent=1,
                                     random_state =45, nthread = -1 , booster = 'gbtree')`

We also tried linear and polynomial regression, also other boosting algos but xgboost gav low rmse value which when compared to mean - si index gives a very low value indicating our model is good and its r2 score was closest to 1 than all of them.


## Part 2:

**ARIMA** is a simple stochastic time series model that we can use to train and then forecast future time points. ARIMA can capture complex relationships as it takes error terms and observations of lagged terms. These models rely on regressing a variable on past values.
We used ARIMA because ARIMA models are applied in some cases where data show evidence of non-stationarity, where an initial differencing step (corresponding to the "integrated" part of the model) can be applied one or more times to eliminate the non-stationarity.
It helped us to find the optimized values of p,d,q for which “AIC” values were low and using them we trained the model. Since past data was short, predictions were satisfactory.



### Contributors

##### 1. Mitanshu Pawan Bhoot
Vishwakarma Institue of Technology<br/>
mitanshu.bhoot18@vit.edu<br/>

##### 2. Tarun Medtiya
Vishwakarma Institue of Technology<br/>
tarun.medtiya18@vit.edu<br/>

##### 3. Saurabh Chandrakant Rane
Vishwakarma Institue of Technology<br/>
saurabh.rane18@vit.edu<br/>

##### 4. Sakshi Jitendra Oswal
Vishwakarma Institue of Technology<br/>
sakshi.oswal18@vit.edu<br/>

