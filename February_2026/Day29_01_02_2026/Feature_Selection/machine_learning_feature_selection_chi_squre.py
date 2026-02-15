from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2, SelectKBest

x, y = load_iris(return_X_y=True)
print(x)
print('-----------------------------')
print(y)
selector = SelectKBest(score_func=chi2, k=2)
x_new = selector.fit_transform(x,y)

print(x.shape)
print(x_new.shape)

selected_col_index = selector.get_support(indices=True)
print(selected_col_index)