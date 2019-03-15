#-*- coding=utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold
#变量晒选
#随机森林重要度
def feature_importance(X,y):    
    feat_labels = X.columns[0:]
    forest = RandomForestClassifier(n_estimators=200, max_depth= 10,random_state=0, n_jobs=-1)
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]),
            importances[indices],
            color='lightblue',
            align='center')
    plt.xticks(range(X.shape[1]),
               pd.DataFrame(X).columns[indices],
               rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
#    plt.savefig("importance.png")
    plt.show()
    
#boruta feature selection
def cal_boruta(df,target,n=50):
    y = df[target]                          
    X = df.drop([target], axis=1).values
    y = y.ravel()
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)
    feat_selector = BorutaPy(rf, n_estimators='auto', max_iter=n, verbose=2, random_state=1)
    feat_selector.fit(X, y)
    feature_df = pd.DataFrame(df.drop([target], axis=1).columns.tolist(), columns=['features'])
    feature_df['rank']=feat_selector.ranking_
    feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)
    return feature_df

#RFE feature selection (建议用树模型)
def cal_RFE(df,target,cls,cv = 3, scoring = ‘roc_auc'):
    y = df[target]
    X = df.drop([target], axis=1)
    estimator = cls
    selector = RFECV(estimator=estimator, cv=StratifiedKFold(cv), scoring = scoring)
    selector.fit(X, y)
    feature_df_RFECV = pd.DataFrame(frame_sampling.drop(['loan_status_recode'], axis=1).columns.tolist(), columns=['features'])
    feature_df_RFECV['rank']=selector.ranking_
    feature_df_RFECV = feature_df_RFECV.sort_values('rank', ascending=True).reset_index(drop=True)
    return feature_df_RFECV
