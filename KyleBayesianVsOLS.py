import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

plt.style.use("seaborn-whitegrid")

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 0)
pd.set_option('display.expand_frame_repr', False)

# folder_path = "C:\\Users\\Arman\\Desktop\\IC business school\\AlgoTradingTutorial"
# cr_dir = os.getcwd()
# os.chdir(folder_path)

# Comparison of two blogsl:
# https://www.richard-stanton.com/2021/06/07/sequential-bayesian-regression.html
#  Kyle Model: https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/cfm-imperial-institute-of-quantitative-finance/events/Lillo-Imperial-Lecture1.pdf


#  Generating some data for Kyle model
# How does the MM learned Lambda if they play many times


# Informed trader: IT
# Noise Trader: NT
# Market Maker: MM
# Price S_T ~ N(S_0,sigma_0)
# NT trade size U ~ N(0,sigma_U)
# Q the optimal trade size of the IT
# Q = (S_T-mu)/(2*lambda_kyle)
# S* = mu + lambda_kyle (Q+U)

S_0 = 1
sigma_0 = 0.5
sigma_U = 1


# Initial guess
mu_0 =0.5
lambda_kyle_0 = 0.5


n_games = 10_000
# setting the random state
np.random.seed(42)
# sampling the True price and NT volume

S_T = np.random.normal(S_0, sigma_0, size=(n_games,1))
U = np.random.normal(0, sigma_U, size=(n_games,1))

# adding the intercept
# S_T = np.hstack([np.ones(shape=(n_games, 1)), S_T])
# setting initial guess for mu and lambda
mu_true = S_0
lambda_kyle_true = 0.5 * sigma_0/  sigma_U
Q = (S_T - S_0)/(2*lambda_kyle_true)
V = Q + U

S_star_true = S_0 +lambda_kyle_true * V




idx = np.floor(np.linspace(0, n_games, num=200))[1:]
time_iter = []

params_mu_lambda_kyle = [np.array([mu_0,lambda_kyle_0])]
params_std = [np.array([np.nan,np.nan])]

t0 = time.process_time()

for end_idx in idx:
    # use the last learned coefficients
    mu_0, lambda_kyle_0 = params_mu_lambda_kyle[-1]
    S_T_obsereved_n = S_T[: int(end_idx)]
    U_obsereved_n = U[: int(end_idx)]

    Q_obsereved_n = (S_T_obsereved_n - mu_0) / (2 * lambda_kyle_0)
    V_obsereved_n = Q_obsereved_n + U_obsereved_n

    # adding intercept
    X = sm.add_constant(V_obsereved_n)
    y = S_T_obsereved_n

    model = sm.OLS(y, X).fit()

    params_mu_lambda_kyle.append(model.params)
    params_std.append(np.sqrt(np.diag(model.normalized_cov_params)))

    time_iter.append(time.process_time() - t0)


params_mu_lambda_kyle = pd.DataFrame(params_mu_lambda_kyle, index= np.array([0] +  list(idx)), columns=['$\mu$','$\lambda_{Kyle}$'])
params_std = pd.DataFrame(params_std, index= np.array([0] +  list(idx)), columns=['$\mu$','$\lambda_{Kyle}$'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(idx, time_iter, label="statsmodels")
ax.set_ylabel("Time taken (s)")
ax.set_xlabel("No. training rows")
plt.legend()
title = 'Time on Learning Kyle Model Coefficients by Sequential OLS.'
title += '\n$S_T=N(S_0,\sigma_0^2)$, $U=N(0,\sigma_U^2)$ and we do the regression on $S^* =E[S_T|V] = \mu +\lambda_{Kyle}V $'
title += f"\nWe set true $\mu=S_0={S_0}$, and "+ "$\lambda_{Kyle}=0.5\sigma_0/\sigma_U"  + f"={lambda_kyle_true}$"
plt.suptitle(title)
plt.tight_layout()
plt.savefig('Time on Learning Kyle Model Coefficients by Sequential OLS'.replace(' ','') + '.pdf',dpi=300)
plt.close()


fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
params_mu_lambda_kyle.plot(ax=ax[0])
params_std.plot(ax=ax[1])
ax[0].set_xlabel("No. training rows")
ax[0].set_ylabel("Coefficient means")
ax[1].set_ylabel("Coefficient standard dev.")
ax[0].set_ylim((0,1.1))
ax[1].set_ylim((0,0.2))
title = 'Learning Kyle Model Coefficients by Sequential OLS.'
title += '\n$S_T=N(S_0,\sigma_0^2)$, $U=N(0,\sigma_U^2)$ and we do the regression on $S^* =E[S_T|V] = \mu +\lambda_{Kyle}V $'
title += f"\nWe set true $\mu=S_0={S_0}$, and "+ "$\lambda_{Kyle}=0.5\sigma_0/\sigma_U" + f"={lambda_kyle_true}$"
plt.suptitle(title)
plt.tight_layout()
plt.savefig('Learning Kyle Model Coefficients by Sequential OLS'.replace(' ','') + '.pdf',dpi=300)
plt.close()

# Sequential bayesian regression


# Another way of approaching this problem is with sequential Bayesian regression. This method follows Bayes theorem, where we have a prior distribution or estimates of our regression coefficients. We then update those prior distribution with data we observe to get a posterior distribution. We can then consider those posterior distributions as our new prior distributions and repeat the process.
#
# Linear regression produces a multivariate normal distribution over the resulting coefficient estimates. The conjugate prior to this is also a multivariate normal distribution prior. As such we can formulate an analytical expression for the Bayes rule posterior update.
#
# The update rules I used were taken from: https://cedar.buffalo.edu/~srihari/CSE574/Chap3/3.4-BayesianRegression.pdf

# We can build this as a class withi a similar API to sklearn models:
class BayesLinearRegressor:
    def __init__(
        self, number_of_features, mean=None, cov=None, alpha=1e6, beta=1
    ):
        # prior distribution on weights
        if mean is None:
            self.mean = np.array([[0.5] * (number_of_features)], dtype=np.float).T

        if cov is None:
            self.cov = alpha * np.identity(number_of_features)
            self.cov_inv = np.linalg.inv(self.cov)

        self.beta = beta  # process noise

    def fit(self, x, y):
        return self.update(x, y)

    def update(self, x, y):
        """
        Perform a bayesian update step
        """
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        if len(y.shape) == 1:
            y = y[:, np.newaxis]

        # update state of covariance and means
        cov_n_inv = self.cov_inv + self.beta * x.T @ x
        cov_n = np.linalg.inv(cov_n_inv)
        mean_n = cov_n @ (self.cov_inv @ self.mean + self.beta * x.T @ y)

        self.cov_inv = cov_n_inv
        self.cov = cov_n
        self.mean = mean_n

    def predict(self, x):
        mean = x @ self.mean
        scale = np.sqrt(np.sum(x @ self.cov @ x.T, axis=1))
        return mean, scale

    @property
    def coef_(self):
        return self.mean

    @property
    def scale_(self):
        return np.sqrt(np.diag(self.cov))

# We can train the model as follows. We use numpy testing to confirm the coefficient we get are equal to those of statsmodels.

mu_0, lambda_kyle_0 = params_mu_lambda_kyle.iloc[0,0],params_mu_lambda_kyle.iloc[0,1]
S_T_obsereved_n = S_T[: int(n_games)]
U_obsereved_n = U[: int(n_games)]

Q_obsereved_n = (S_T_obsereved_n - mu_0) / (2 * lambda_kyle_0)
V_obsereved_n = Q_obsereved_n + U_obsereved_n

# Simplifying it in terms of S_T and U: S_T = \mu_0 + \lambda_kyle * V
# is the same as S_T = \mu_0 + 2*\lambda_kyle * U where V= Q+U and Q= (S_T-\mu)/(2*\lambda_kyle)


Q_obsereved_n = (S_T_obsereved_n - mu_0) / (2 * lambda_kyle_0)
V_obsereved_n = Q_obsereved_n + U_obsereved_n

# adding intercept
X = sm.add_constant(V_obsereved_n)
y = S_T_obsereved_n
model = sm.OLS(y, X).fit()

mean_list = np.array([[mu_0, lambda_kyle_0]], dtype=np.float).T


bayes_linear_regression = BayesLinearRegressor(X.shape[1])
bayes_linear_regression.mean = np.array([[mu_0, lambda_kyle_0]], dtype=np.float).T
bayes_linear_regression.fit(X, y)
# np.testing.assert_array_almost_equal(
#     bayes_linear_regression.coef_, params_mu_lambda_kyle.tail(1).transpose().to_numpy()
# )
# np.testing.assert_array_almost_equal(
#     bayes_linear_regression.scale_, params_std.tail(1).to_numpy().flatten()
# )


bayes_linear_regression = BayesLinearRegressor(X.shape[1])

time_iter_seq = []
params_mu_seq_bayes = [np.array([mu_0,lambda_kyle_0])]
params_std_seq_bayes = [np.array([np.nan,np.nan])]
t0 = time.process_time()

for i, end_idx in enumerate(idx):
    if i > 0:
        start_idx = int(idx[i - 1])
    else:
        start_idx = 0
    mu_i, lambda_kyle_i = params_mu_seq_bayes[-1]
    # use the last learned coefficients
    S_T_obsereved_n = S_T[start_idx : int(end_idx)]
    U_obsereved_n = U[start_idx : int(end_idx)]

    Q_obsereved_n = (S_T_obsereved_n - mu_i) / (2 * lambda_kyle_i)
    V_obsereved_n = Q_obsereved_n + U_obsereved_n

    # adding intercept
    X = sm.add_constant(V_obsereved_n)
    y = S_T_obsereved_n

    bayes_linear_regression.update(
        X,
        y,
    )
    time_iter_seq.append(time.process_time() - t0)

    params_mu_seq_bayes.append(bayes_linear_regression.coef_.flatten())
    params_std_seq_bayes.append(bayes_linear_regression.scale_)

params_mu_seq_bayes = pd.DataFrame(params_mu_seq_bayes, index= np.array([0] +  list(idx)), columns=['$\mu$','$\lambda_{Kyle}$'])
params_std_seq_bayes = pd.DataFrame(params_std_seq_bayes, index= np.array([0] +  list(idx)), columns=['$\mu$','$\lambda_{Kyle}$'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(idx, time_iter, label="Sequential Statsmodels Linear regression")
ax.plot(idx, time_iter_seq, label="Sequential Bayesian Learning")
# ax.plot(idx, np.cumsum(time_iter_seq), label="cumulative_sequential")
ax.set_ylabel("Time taken (s)")
ax.set_xlabel("No. training rows")
title = 'Time on Learning Kyle Model Coefficients by Sequential OLS regression vs Bayesian Regression.'
title += '\n$S_T=N(S_0,\sigma_0^2)$, $U=N(0,\sigma_U^2)$ and we do the regression on $S^* =E[S_T|V] = \mu +\lambda_{Kyle}V $'
title += f"\nWe set true $\mu=S_0={S_0}$, and "+ "$\lambda_{Kyle}=0.5\sigma_0/\sigma_U"  + f"={lambda_kyle_true}$"
plt.suptitle(title)
plt.legend()
plt.tight_layout()
plt.savefig('Time on Learning Kyle Model Coefficients by Sequential OLS regression vs Bayesian Regression'.replace(' ','') + '.pdf',dpi=300)
plt.close()



fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
params_mu_seq_bayes.plot(ax=ax[0])
params_std_seq_bayes.plot(ax=ax[1])
ax[0].set_xlabel("No. training rows")
ax[0].set_ylabel("Coefficient means")
ax[1].set_ylabel("Coefficient standard dev.")
ax[0].set_ylim((0,1.1))
ax[1].set_ylim((0,0.2))
title = 'Learning Kyle Model Coefficients by Bayesian Regression.'
title += '\n$S_T=N(S_0,\sigma_0^2)$, $U=N(0,\sigma_U^2)$ and we do the regression on $S^* =E[S_T|V] = \mu +\lambda_{Kyle}V $'
title += f"\nWe set true $\mu=S_0={S_0}$, and "+ "$\lambda_{Kyle}=0.5\sigma_0/\sigma_U"  + f"={lambda_kyle_true}$"
plt.suptitle(title)
plt.legend()
plt.tight_layout()
plt.savefig('Learning Kyle Model Coefficients by Bayesian Regression'.replace(' ','') + '.pdf',dpi=300)
plt.close()


# ### Posteriors vs Priors One of the benefits of using bayesian linear regression is the ability to apply prior distributions on the model coefficients.
# To demonstrate this we use a prior with a much smaller variance, as such it is no longer uninformed.
# We fit the model and plot the pdf of the prior and posterior.
# The posteriors evidently converge to the true coefficients and have a tight distribution.

bayes_linear_regression = BayesLinearRegressor(X.shape[1], alpha=0.5)
prior_mu = bayes_linear_regression.coef_
prior_std = bayes_linear_regression.scale_
# bayes_linear_regression.fit(x, y)


def norm_max(x):
    return x / x.max()


x_range = np.linspace(-2, 2, num=1000)
fig, ax = plt.subplots(nrows=2, figsize=(10, 6))
for idx in range(2):
    ax[idx].plot(
        x_range,
        norm_max(norm.pdf(x_range, loc=prior_mu[idx], scale=prior_std[idx])), label="Prior Distribution"
    )
    ax[idx].plot(
        x_range,
        norm_max(
            norm.pdf(
                x_range,
                loc=params_mu_seq_bayes.iloc[-1, idx],
                scale=params_std_seq_bayes.iloc[-1, idx],
            )
        ), label="Posterior Distribution"
    )
    ax[idx].set_ylabel(f"$~P({params_mu_seq_bayes.columns.to_list()[idx].replace(r'$','')})$")
    # title += f"\nWe set true $\mu=S_0={S_0}$, and "+ "$\lambda_{Kyle}=0.5\sigma_0/\sigma_U"  + f"={lambda_kyle_true}$"
    ax[idx].set_title(params_mu_seq_bayes.columns.to_list()[idx])

title = 'Posteriors vs Prior\nWe have tight Posteriors, which is pleasurable'
# title += f"\nWe set true $\mu=S_0={S_0}$, and "+ "$\lambda_{Kyle}=0.5\sigma_0/\sigma_U"  + f"={lambda_kyle_true}$"
plt.suptitle(title)
plt.legend()
plt.tight_layout()
plt.savefig('Posteriors vs Prior'.replace(' ','') + '.pdf',dpi=300)
plt.close()

# Prediction uncertainty
# The distribution of our coefficients gives us a distribution for our model predictions as well.
#
# The predict method of the BayesLinearRegressor class returns a standard deviation for each point. We can then plot a few
# of those points with this confidence shaded. This only represent epistemic uncertainty - i.e. uncertainty from our model coefficients,
# not uncertainty from the data generating process.


bayes_linear_regression = BayesLinearRegressor(X.shape[1])
bayes_linear_regression.fit(X, y)
x_volume = X[:100].T[1]
sort_id = np.argsort(x_volume.flatten())
x_volume = x_volume[sort_id]
x_volume =x_volume.T
X_x_volume = sm.add_constant(x_volume.T)


pred_mu, pred_scale = bayes_linear_regression.predict(X_x_volume)


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_volume, pred_mu.flatten(), ".")

ax.fill_between(
    x_volume,
    (pred_mu.flatten() - pred_scale).flatten(),
    (pred_mu.flatten() + pred_scale).flatten(),
    alpha=0.3,
)
ax.set_xlabel("Net Trading Volume $V=Q+U$")
ax.set_ylabel("$S^*$ Predicted execution Price")
title = '$S^*$ Prediction uncertainty'
title += f"\n$\mu=S_0={S_0}$, and "+ "$\lambda_{Kyle}=0.5\sigma_0/\sigma_U"  + f"={lambda_kyle_true}$"
plt.suptitle(title)
plt.ylim((-4,4))
plt.tight_layout()
plt.savefig('Prediction uncertainty'.replace(' ','') + '.pdf',dpi=300)
plt.close()


# os.chdir(cr_dir)


