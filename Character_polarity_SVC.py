import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.svm import SVC
import sklearn.metrics as mt
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
db = pymysql.connect(
    user='web_user',
    passwd='cau523.',
    port=3308,
    host='localhost',
    db='emotions',
    charset='utf8'
)
cursor = db.cursor(pymysql.cursors.DictCursor)

cursor.execute("select ID, 긍정_감정1, 긍정_감정2, 긍정_감정3, 긍정_감정4, 긍정_감정5,"
               "부정_감정1, 부정_감정2, 부정_감정3, 부정_감정4, 부정_감정5,"
               "개방성_평균, 성실성_평균, 신경증_평균, 외향성_평균, 친화성_평균,"
               "pos_keyword, neg_keyword from svm_test where pos_keyword is not null and neg_keyword is not null")

data = cursor.fetchall()

list_x=[]
list_y=[]
list_char = []
for i in data:
    list_char.append([float(i['개방성_평균']), float(i['성실성_평균']), float(i['신경증_평균']), float(i['외향성_평균']), float(i['친화성_평균'])])
    list_char.append([float(i['개방성_평균']), float(i['성실성_평균']), float(i['신경증_평균']), float(i['외향성_평균']), float(i['친화성_평균'])])
    list_x.append(i['pos_keyword'])
    list_y.append(0)
    list_x.append(i['neg_keyword'])
    list_y.append(1)
print(list_char)
transformer = CountVectorizer()
list_x = transformer.fit_transform(list_x).toarray()
pca = PCA(n_components=1)
list_x = pca.fit_transform(list_x)
print("list_x1:",list_x)

list_char = np.array(list_char)
list_char = pca.fit_transform(list_char)
print("list_char:",list_char)
list_x2 = []
list_x2 = np.array(list_x2)

list_x2 = np.hstack((list_x, list_char))

print("list_x2:",list_x2)

sc = StandardScaler()
sc.fit(list_x2)
list_x = sc.transform(list_x2)
X_train, X_test, y_train, y_test = train_test_split(list_x, list_y, test_size = 0.2, random_state = 0)

svm_model = SVC(kernel='rbf')

parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100],
             'gamma':[0.001, 0.01, 0.1, 1, 10, 25, 50, 100]}


grid_svm = GridSearchCV(svm_model, param_grid = parameters, cv = 5)

grid_svm.fit(X_train, y_train)

result = pd.DataFrame(grid_svm.cv_results_['params'])
result['mean_test_score'] = grid_svm.cv_results_['mean_test_score']
b = result.sort_values(by='mean_test_score', ascending=False)

print(b)

print("best parameters:",grid_svm.best_params_)

C = grid_svm.best_params_['C']
gamma = grid_svm.best_params_['gamma']

svm_model = SVC(kernel='rbf', C=C,gamma=gamma)

scores = cross_val_score(svm_model, list_x, list_y, cv = 5)

a = pd.DataFrame(cross_validate(svm_model, list_x, list_y, cv = 5))
print(a)
print('교차검증 평균: ', scores.mean())


svm_model.fit(X_train, y_train)  # SVM 분류 모델 훈련

y_pred = svm_model.predict(X_test)  # 테스트

print("예측된 라벨:", y_pred)
print("ground-truth 라벨:", y_test)

print("prediction accuracy: {:.2f}".format(np.mean(y_pred == y_test)))  # 예측 정확도

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


X0, X1 = X_train[:,0], X_train[:,1]
xx, yy = make_meshgrid(X0, X1)


Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])


Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)


plt.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title('E_character_polarity')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
