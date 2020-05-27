import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error, explained_variance_score, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabulate import tabulate


class Pipeline:

    def __init__(self):
        pass

    def mean_absolute_percentage_error(self,y_true, y_pred):
        ## Note: does not handle mix 1d representation
        #if _is_1d(y_true):
        #    y_true, y_pred = _check_1d_array(y_true, y_pred)
        count = 0
        sum = 0
        if 0 in y_true:
            for i in range(len(y_true)):
                if y_true[i] != 0:
                    count += 1
                    sum += abs(y_true[i] - y_pred[i]) / y_true[i]
            return sum/count
        else:
            return np.mean(np.abs((y_true - y_pred) / y_true))


    def plot_learning_curve2(X_train,y_train,model_):

        model = model_[0]
        modelname = model_[1]

        # divide into cross validation and training set
        #np.random.shuffle(X_train)
        indicator = int(1*len(X_train)/5)
        X_cv_set, X_training_set = X_train[:indicator, :], X_train[indicator:, :]
        y_cv_set, y_training_set = y_train[:indicator, :], y_train[indicator:, :]

        #loop!
        number_of_samples = [int((i+1)*len(X_training_set)/5) for i in range(5)]
        #print('number_of_samples',number_of_samples)
        training_set_error = []
        cv_set_error = []
        for sample in number_of_samples:
            model.fit(X_training_set[:sample, :],y_training_set[:sample, :].ravel())
            y_pred_train = model.predict(X_training_set[:sample, :])
            y_pred_cv = model.predict(X_cv_set)
            training_set_error.append(mean_absolute_error(y_training_set[:sample, :], y_pred_train))
            cv_set_error.append(mean_absolute_error(y_cv_set, y_pred_cv))
        #print(number_of_samples,'\n',training_set_error,'\n',cv_set_error)
        plt.plot(number_of_samples, training_set_error)
        plt.plot(number_of_samples, cv_set_error)
        plt.title(modelname)

    def runpipeline(self,target,technique):
        self.target = target
        self.technique = technique

        df = pd.read_csv('data.csv')

        # features and target
        X_fields = ['VM_size','pdr','working_set_pages','wse','nwse','mwpp', #vm fitures
                   'vm_perf_instructions','max-bandwidth',#Source host (SRC_vm_cpu or SRC_vm_cpu_baseline)
                    'SRC_cpu_system','SRC_cpu_user','DST_cpu_system','DST_cpu_user','SRC_net_manage_ifutil','DST_net_manage_ifutil','SRC_vm_mem_baseline',
                    #Src + dest host (not sure about (NET.UTIL)) ---- dst mem utilization not found! what is memory percentage?
                    'non_working_set_pages','RPTR', 'THR_benefit', 'DLTC_benefit', 'ewss', 'enwss'#composed (nwss = non_working_set_pages) , 'DTC_benefit', 'POST_benefit'
                    ]
        y_fields = self.target#['MEM']


        tech_uniq = df.technique.unique()
        i = 6
        #print(tech_uniq[i],self.technique)
        df_tech =df.loc[df['technique'] == self.technique].copy() #tech_uniq[i]
        #print(tech_uniq,tech_uniq[i])
        # shuffling and split
        mask = np.random.rand(len(df_tech)) < 0.8
        train = df_tech[mask]
        X_train_raw = train[X_fields].values
        y_train = train[y_fields].values
        test = df_tech[~mask]
        X_test_raw = test[X_fields].values
        y_test = test[y_fields].values.ravel()

        X_raw = df_tech[X_fields].values
        y = df_tech[y_fields].values

        # normalization
        standard_scaler = StandardScaler()
        X_train =standard_scaler.fit_transform(X_train_raw)
        X_test = standard_scaler.fit_transform(X_test_raw)
        X = standard_scaler.fit_transform(X_raw)

        # model and train
        models_list = []
        # linear regression
        regressor = LinearRegression(fit_intercept=True)
        regressor.fit(X_train, y_train.ravel()) # training the algorithm
        models_list.append((regressor,"lin-reg"))
        '''
        # Ridge regression
        reg = Ridge(fit_intercept=True,alpha=.5)
        reg.fit(X_train, y_train.ravel()) # training the algorithm
        models_list.append((reg,"lin-Ridge"))
        '''

        '''
        # random Forest
        model = RandomForestRegressor(n_estimators=10, max_features=2)
        model.fit(X_train, y_train)
        models_list.append((model,"RF"))
        '''
        '''
        #SVR
        print('svr')
        model = svm.SVR(kernel='rbf',C=10)
        model.fit(X_train, y_train.ravel())
        models_list.append((model,"SVR"))
        '''
        '''
        # MLP
        print('mlp')
        mlp = make_pipeline(StandardScaler(),
                            MLPRegressor(hidden_layer_sizes=(10,10,10),
                                         tol=1e-2, max_iter=300, random_state=1))
        mlp.fit(X_train, y_train.ravel())
        models_list.append((mlp,"mlp"))
        '''

        # predict, error and plot loop for all models
        asnwer = []
        for model_ in models_list:
            #print(model_[1])
            y_pred = model_[0].predict(X_test)
            #print(y_test.shape, y_pred.shape)
            my_error = 1#sum(abs(y_test-y_pred)/y_test)/y_test.shape[0]
            MSE = mean_squared_error(y_test, y_pred)
            MAE = mean_absolute_error(y_test, y_pred)
            asnwer.append(y_test.mean())
            asnwer.append(y_pred.mean())
            asnwer.append(abs(y_pred.mean()-y_test.mean()))
            asnwer.append(MAE)

            MAPE = self.mean_absolute_percentage_error(y_test, y_pred)
            asnwer.append(MAPE)

            EVS = explained_variance_score(y_test, y_pred)
            R2 = r2_score(y_test, y_pred)

            #score = model_[0].score(y_test, y_pred)
            #print('model: ',model_[1],'\n my_error ', my_error,'\n MSE ', MSE, '\n MAE ', MAE, '\n EVS ', EVS, '\n R2', R2, '\n MAPE', MAPE)#, '\n score', score)
            #plot_learning_curve2(X_train, y_train, model_)
            #plot_learning_curve(model_[0], model_[1], X_train, y_train)
            #plt.show()

        return [(a) for a in asnwer ]
        #y_test.mean()----y_pred.mean()----abs(y_pred.mean()-y_test.mean())----MAE----MAPE



        '''
        #visualize
        
        df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
        df1 = df.head(25)
        df1.plot(kind='bar',figsize=(16,10))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()
        
        plt.scatter(X_test[:,2], y_test,  color='gray')
        plt.scatter(X_test[:,2], y_pred, color='red', linewidth=2)
        plt.show()
        '''



    def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        axes : array of 3 axes, optional (default=None)
            Axes to use for plotting the curves.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the dtype is float, it is regarded as a
            fraction of the maximum size of the training set (that is determined
            by the selected validation method), i.e. it has to be within (0, 1].
            Otherwise it is interpreted as absolute sizes of the training sets.
            Note that for classification the number of samples usually have to
            be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,#,scoring="neg_mean_squared_error"
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt

'''
p = Pipeline()

result = []

ex = 10
for i in range(ex):
    ans = p.runpipeline(target='TT',technique='PRE')
    # y_test.mean()----y_pred.mean()----abs(y_pred.mean()-y_test.mean())----MAE----MAPE
    result.append(ans)
    print(ans)

print(result)
print(tabulate(result,headers=['y_test.mean()','y_pred.mean()','abs(y_pred.mean()-y_test.mean())','MAE','MRE']))

mre=0
mae=0
for r in result:
    mre += r[-1]
    mae += r[3]
mre/=ex
mae/=ex
print("mean of MRE",mre)
print("mean of MAE",mae)
'''