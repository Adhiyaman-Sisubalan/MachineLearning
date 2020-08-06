# -*- coding: utf-8 -*-

import timeit as timeit

import HeaderFile as hf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def SetAxisProperties(ax, title):
    ax.set_title(title)
    ax.set_xlabel("EP (Actual)")
    ax.set_ylabel("EP (Predictions)")


df, features, target = hf.ReadData("power-plant.csv")
df_describe = df.describe()
df_head = df.head()
df_tail = df.tail()
df_corr = df.corr()

hf.plt.figure()
ax_corr = hf.sns.heatmap(df_corr, annot=True, cbar_kws={"shrink": 0.5}, cmap="jet", fmt='g', square=True)
ax_corr.set_title("Correlations Between Features and Target")
ax_corr.xaxis.set_label_position("top")
ax_corr.xaxis.set_ticks_position("top")
bottom, top = ax_corr.get_ylim()
ax_corr.set_ylim(bottom + 0.5, top - 0.5)

hf.pd.plotting.scatter_matrix(df, figsize=(15, 15), diagonal='kde')
hf.plt.suptitle("Scatter Matrix of Features and Target", size=20)

hf.plt.figure(figsize=(15, 15))
hf.plt.suptitle("Plots of Target vs Individual Features")
ax_at_ep = hf.plt.subplot(221)
ax_at_ep.set_title("Ambient Temperature")
hf.sns.scatterplot(x="AT", y="EP", data=df, color="red")
ax_ev_ep = hf.plt.subplot(222)
ax_ev_ep.set_title("Exhaust Vacuum")
hf.sns.scatterplot(x="EV", y="EP", data=df, color="green")
ax_ap_ep = hf.plt.subplot(223)
ax_ap_ep.set_title("Ambient Pressure")
hf.sns.scatterplot(x="AP", y="EP", data=df, color="blue")
ax_rh_ep = hf.plt.subplot(224)
ax_rh_ep.set_title("Relative Humidity")
hf.sns.scatterplot(x="RH", y="EP", data=df, color="purple")

X, y = hf.TransformData(features, target, resample=False, scale=True, pca=False, pca_plot=False)

number_iterations = 4
cv_scores_mean = hf.np.zeros(number_iterations)
cv_scores_std = hf.np.zeros(number_iterations)
for i in range(number_iterations):
    X_data = X.iloc[:, 0:i+1]
    lin_reg = LinearRegression()
    scores = hf.cross_val_score(estimator=lin_reg, X=X_data, y=y, cv=10, error_score=0, n_jobs=-1, scoring="neg_mean_squared_error")
    cv_scores_mean[i] = scores.mean()
    cv_scores_std[i] = scores.std()

hf.Plot2D("line", hf.np.arange(4), [[-y for y in cv_scores_mean]],
          title="Average Mean Squared Errors with Increasing Number of Features",
          xlabel="Number of Features",
          ylabel="MSE Scores",
          xticks={"ticks": hf.np.arange(4), "labels": [1, 2, 3, 4]},
          yticks={"ticks": hf.np.arange(18, 32, step=2)})

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lin_reg = LinearRegression()
start_time = timeit.default_timer()
lin_reg_model = lin_reg.fit(X_train, y_train)
end_time = timeit.default_timer()
elapsed_time = end_time - start_time
b = lin_reg.intercept_
w = lin_reg.coef_
y_hat = lin_reg.predict(X_train)
r2_hat = r2_score(y_train, y_hat)
y_pred = lin_reg.predict(X_test)
r2_pred = r2_score(y_test, y_pred)

hf.plt.figure()
hf.plt.suptitle("Predictions vs Actual Data")
ax_left = hf.plt.subplot(121)
hf.sns.regplot(y_train, y_hat, label="hello")
ax_right = hf.plt.subplot(122, sharex=ax_left, sharey=ax_left)
hf.sns.regplot(y_test, y_pred)
SetAxisProperties(ax_left, r"Train Data (${r^2}\approx{%.3f}$)" % (r2_hat))
SetAxisProperties(ax_right, r"Test Data (${r^2}\approx{%.3f}$)" % (r2_pred))
