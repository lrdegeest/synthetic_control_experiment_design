from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import seaborn as sns

import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from sklearn.base import BaseEstimator, RegressorMixin

import cvxpy as cp
from joblib import Parallel, delayed

from pandas.api.types import is_string_dtype
from tqdm import tqdm

from scipy.interpolate import interp1d, PchipInterpolator

from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull


def get_mde(power_df, target_power=0.8, smooth=True, n_points=300):
    """
    Compute the Minimum Detectable Effect (MDE) for both positive and negative sides
    using interpolation of the power curve.

    Parameters
    ----------
    power_df : pd.DataFrame
        Must contain columns ['effect', 'reject'] where 'reject' = mean power.
    target_power : float, optional
        Desired power threshold (default 0.8).
    smooth : bool, optional
        If True, apply smooth interpolation (PCHIP). Otherwise use raw grid.
    n_points : int, optional
        Number of interpolation grid points (default 300).

    Returns
    -------
    mde_pos : float or np.nan
        Minimum positive detectable effect size achieving target power.
    mde_neg : float or np.nan
        Minimum negative detectable effect size achieving target power.
    """
    power_curve = (
        power_df.groupby('effect', as_index=False)['reject'].mean()
        .sort_values('effect')
        .reset_index(drop=True)
    )

    effects = power_curve['effect'].values
    power_vals = power_curve['reject'].values

    if smooth and len(effects) > 2:
        f_interp = PchipInterpolator(effects, power_vals)
        x_fine = np.linspace(min(effects), max(effects), n_points)
        y_fine = np.clip(f_interp(x_fine), 0, 1)
    else:
        x_fine, y_fine = effects, power_vals

    pos_mask = x_fine >= 0
    neg_mask = x_fine <= 0

    mde_pos = np.nan
    mde_neg = np.nan

    if np.any(y_fine[pos_mask] >= target_power):
        idx_pos = np.argmax(y_fine[pos_mask] >= target_power)
        mde_pos = x_fine[pos_mask][idx_pos]

    if np.any(y_fine[neg_mask] >= target_power):
        idx_neg = np.argmax(y_fine[neg_mask][::-1] >= target_power)
        mde_neg = x_fine[neg_mask][::-1][idx_neg]

    return mde_pos, mde_neg
    

def plot_power_comparison(
    *dfs_with_labels,
    title: str = "Power Comparison",
    figsize: tuple = (7, 5),
    ylabel: str = "Power",
    xlabel: str = "Effect Size (%)",
    target_power: float = 0.8,
    subtitle: str = None,
    return_summary: bool = False,
):
    """
    Plot power curves for an arbitrary number of designs.

    Parameters
    ----------
    *dfs_with_labels : tuple
        Any number of (label, DataFrame) pairs. Each DataFrame must contain
        'effect' and 'reject' columns
        Example: ('forward', power_forward), ('random', power_random)
    title : str
        Main title of the plot
    figsize : tuple
        Figure size.
    ylabel, xlabel : str
        Axis labels.
    target_power : float
        Draw a horizontal line at this power level.
    subtitle : str, optional
        Subtitle shown above the plot.
    return_summary : bool
        If True, returns a summary DataFrame with mean power by effect and design.
    """
    all_dfs = []
    for label, df in dfs_with_labels:
        df_copy = df.copy()
        df_copy["design"] = label
        all_dfs.append(df_copy)
    power_all = pd.concat(all_dfs, ignore_index=True)

    power_summary = (
        power_all.groupby(["design", "effect"], as_index=False)
        .agg(power=("reject", "mean"))
    )

    plt.figure(figsize=figsize)
    for design_name, df_design in power_summary.groupby("design"):
        plt.plot(
            df_design["effect"] * 100,
            df_design["power"],
            label=design_name.title(),
            lw=2
        )

    plt.axhline(target_power, color="k", ls="--", lw=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=14)
    if subtitle:
        plt.suptitle(subtitle, fontsize=10, y=0.95)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if return_summary:
        return power_summary




def infer_time_freq(times: np.ndarray) -> str:
    """
    Check if data are weekly or daily
    """
    times = pd.Series(pd.to_datetime(times)).sort_values()
    deltas = times.diff().dropna()
    most_common_diff = deltas.mode().iloc[0]
    if most_common_diff == pd.Timedelta(days=1):
        return 'D'
    elif most_common_diff == pd.Timedelta(weeks=1):
        return 'W'
    else:
        raise ValueError(f"Unsupported frequency: {most_common_diff}. Only 'D' or 'W' supported.")            


def simulate_factor_model(
    N=100, T=400, G=5, K=3, convex=True,
    between_cluster_scale=1.0, within_cluster_scale=1.0,
    factor_noise_scale=5.0, idiosyncratic_noise_scale=1.0,
    base_level=100, cluster_random_effect_scale=10.0,
    distribution='normal',
    cluster_center='normal',
    phi_factors=0.8, seed=None
):
    """
    Simulate clustered panel data using a AR(1) k-factor model

    Parameters:
    - N: int, number of units
    - T: int, number of time periods
    - G: int, number of clusters
    - K: int, number of latent factors
    - between_cluster_scale: float, standard deviation for cluster centers (factor loadings)
    - within_cluster_scale: float, standard deviation for unit-level factor loadings
    - factor_noise_scale: float, standard deviation for latent factors random walk increments
    - idiosyncratic_noise_scale: float, standard deviation for idiosyncratic noise
    - base_level: float, global intercept level for Y
    - cluster_random_effect_scale: float, standard deviation for cluster-level random intercepts
    - seed: int or None, random seed for reproducibility

    Returns:
    - Y: (T x N) ndarray, simulated outcome data
    - Ybar: (T x N) ndarray, factor-driven signal (no noise)
    - Lambda: (K x N) ndarray, factor loadings
    - F: (T x K) ndarray, latent factors over time
    - unit_cluster: (N,) ndarray, cluster assignment for each unit (1-indexed)
    - cluster_baselines: (G,) ndarray, random intercepts per cluster
    """
    if seed is not None:
        np.random.seed(seed)

    cluster_size = N // G
    unit_cluster = np.repeat(np.arange(1, G + 1), cluster_size)

    if cluster_center == 'normal':
        cluster_centers = np.random.normal(0, between_cluster_scale, size=(G, K))
    elif cluster_center == 'dirichlet': 
        cluster_centers = np.random.dirichlet([1] * K, size=G) * between_cluster_scale
    
    Lambda = np.zeros((K, N))
        
    # choose an epsilon
    for i in range(N):
        g = unit_cluster[i] - 1
        if distribution == "normal":
            noise = np.random.normal(0, within_cluster_scale, K)
        elif distribution == "lognormal":
            noise = np.random.lognormal(0, within_cluster_scale, K)
        elif distribution == "uniform":
            noise = np.random.uniform(-within_cluster_scale, within_cluster_scale, K)
        elif distribution == "gamma":
            noise = np.random.gamma(shape=2.0, scale=within_cluster_scale, size=K) - 2.0 * within_cluster_scale
        else:
            raise ValueError(f"Unsupported distribution {distribution}")
        Lambda[:, i] = cluster_centers[g] + noise        

    if convex:
        # normalize Lambda to be convex weights (so we have a convex factor model)
        # forces each unit's loadings to sum to one and keeps Y on the same scale across units
        Lambda = Lambda / Lambda.sum(axis=0, keepdims=True)

    # AR(1) factor process 
    F = np.zeros((T, K))
    F[0, :] = np.random.normal(0, factor_noise_scale / np.sqrt(1 - phi_factors**2), size=K)
    for t in range(1, T):
        F[t, :] = phi_factors * F[t - 1, :] + np.random.normal(0, factor_noise_scale, size=K)

    # noise, cluster intercepts
    E = np.random.normal(0, idiosyncratic_noise_scale, size=(T, N))
    cluster_baselines = np.random.normal(base_level, cluster_random_effect_scale, size=G)
    unit_baselines = np.array([cluster_baselines[g - 1] for g in unit_cluster])

    Ybar = F @ Lambda
    Y = base_level + Ybar + E + unit_baselines[np.newaxis, :]

    return Y, Ybar, Lambda, F, unit_cluster, cluster_baselines    


def plot_factor_clusters(Lambda, unit_cluster, three_dim=False, figsize=(9, 6)):
      
    if three_dim:
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        clusters = np.unique(unit_cluster)
        colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))

        for i, cluster_id in enumerate(clusters):
            mask = (unit_cluster == cluster_id)
            ax.scatter(Lambda[0, mask], Lambda[1, mask], Lambda[2, mask],
                       color=colors[i], label=f'Cluster {cluster_id}', alpha=0.7)
        
            ax.set_xlabel('λ₁')
            ax.set_ylabel('λ₂')
            ax.set_zlabel('λ₃')
        plt.legend()
        plt.show()
        
    else:
        
        X = Lambda.T
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        hull = ConvexHull(X_pca)
    
        plt.figure(figsize=figsize)
        clusters = np.unique(unit_cluster)
        colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
    
        for i, cluster_id in enumerate(clusters):
            cluster_mask = (unit_cluster == cluster_id)
            plt.scatter(
                X_pca[cluster_mask, 0],
                X_pca[cluster_mask, 1],
                color=colors[i],
                label=f'Cluster {cluster_id}',
                alpha=0.7
            )
    
        for simplex in hull.simplices:
            plt.plot(X_pca[simplex, 0], X_pca[simplex, 1],
                     color='gray', linestyle='--', lw=2, alpha=0.8)
    
        plt.plot(
            X_pca[hull.vertices[[-1, 0]], 0],
            X_pca[hull.vertices[[-1, 0]], 1],
            color='gray', linestyle='--', lw=2, alpha=0.8
        )
    
        plt.xlabel("Principal Component 1", fontsize=12)
        plt.ylabel("Principal Component 2", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()



def simulate_experiment(Y, 
                        unit_cluster, 
                        treatment_start=350, 
                        relative=False,
                        treatment_effect=5.0, 
                        treated_cluster=None, 
                        treated_units_size=3, 
                        seed=None):
    """
    Simulate an experiment by applying treatment to selected units and aggregating to weekly data

    Parameters:
    - Y: (T x N) ndarray, outcome matrix from factor model
    - unit_cluster: (N,) ndarray, cluster assignment for each unit
    - treatment_start: int, start date of treatment (day index)
    - treatment_effect: float, magnitude of the treatment effect
    - treated_cluster: int or None, cluster to select treated units from. If None, select randomly across all units.
    - treated_units_size: int, number of treated units
    - seed: int or None, random seed for reproducibility

    Returns:
    - weekly_df: DataFrame, aggregated weekly panel data with treatment applied
    - treated_units: list of treated unit IDs (as strings)
    - treatment_start_str: string, treatment start date (ISO format)
    """
    if seed is not None:
        np.random.seed(seed)

    T, N = Y.shape
    base_date = pd.to_datetime('2021-04-01')

    # select treated units (optionally within a cluster)
    pool = np.where(unit_cluster == treated_cluster)[0] if treated_cluster is not None else np.arange(N)
    treated_units = np.random.choice(pool, size=treated_units_size, replace=False)

    # build a long panel
    df_long = (
        pd.DataFrame(Y, columns=np.arange(N))
        .assign(time=np.arange(T))
        .melt(id_vars='time', var_name='unit', value_name='outcome')
    )
    df_long['unit'] = df_long['unit'].astype(int)
    df_long['date'] = base_date + pd.to_timedelta(df_long['time'], unit='D')
    df_long['time'] = df_long['date'].dt.to_period('W').apply(lambda r: r.start_time)

    # merge with unit info
    unit_map = pd.DataFrame({
        'unit': np.arange(N),
        'treated': np.isin(np.arange(N), treated_units),
        'cluster': unit_cluster
    })
    df_long = df_long.merge(unit_map, on='unit')

    # aggregate to weekly level
    weekly_df = (
        df_long
        .groupby(['time', 'unit', 'treated', 'cluster'], as_index=False)
        .agg(outcome=('outcome', 'mean'))
    )

    # compute treatment start week
    start_date = base_date + pd.to_timedelta(treatment_start, unit='D')
    treatment_week = start_date.to_period('W').start_time
    treatment_start_str = str(treatment_week.date())

    # apply treatment effect
    treated_mask = (
        weekly_df['treated'] &
        (weekly_df['time'] >= treatment_week)
    )
    if relative:
        weekly_df.loc[treated_mask, 'outcome'] *= (1 + treatment_effect)
    else:
        weekly_df.loc[treated_mask, 'outcome'] += treatment_effect

    # add post-treatment flag
    weekly_df['post_treatment'] = weekly_df['time'] >= treatment_week

    weekly_df['unit'] = weekly_df['unit'].astype('str')

    return weekly_df, [str(u) for u in treated_units], treatment_start_str



#=======================================
# synthetic control estimator
# need to add inference methods
# mostly just used this to sanity check factor simulator 
#=======================================
class Synth: 
    def __init__(self, df, metric_col, unit_col, time_col,
                 treated_units, treatment_start, solver: str = 'CLARABEL'):
        self.df = df.copy()
        self.metric_col = metric_col
        self.unit_col = unit_col
        self.time_col = time_col
        self.treated_units = [str(u) for u in treated_units]
        self.treatment_start = pd.to_datetime(treatment_start)
        self.weights_ = None
        self.results_ = None
        self.ctrl_units_ = None
        self.solver = solver

    def fit(self):
        """Estimate synthetic control weights via Abadie 2010"""
        df = self.df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        pre_df = df[df[self.time_col] < self.treatment_start]

        treated = pre_df[pre_df[self.unit_col].isin(self.treated_units)]
        control = pre_df[~pre_df[self.unit_col].isin(self.treated_units)]

        y_treat = (
            treated.groupby(self.time_col)[self.metric_col]
            .mean()
            .to_numpy()
        )

        y_ctrl = (
            control.pivot(index=self.time_col, columns=self.unit_col,
                          values=self.metric_col)
            .to_numpy()
        )

        w = cp.Variable(y_ctrl.shape[1], nonneg=True)
        objective = cp.Minimize(cp.sum_squares(y_treat - y_ctrl @ w))
        constraints = [cp.sum(w) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        self.weights_ = w.value
        self.ctrl_units_ = list(control[self.unit_col].unique()) 

        y_hat_pre = y_ctrl @ self.weights_
        ssr = np.sum((y_treat - y_hat_pre)**2)
        sst = np.sum((y_treat - np.mean(y_treat))**2)
        self.pre_r2_ = 1 - ssr / sst if sst > 0 else np.nan

    def predict(self):
        """Generate the counterfactual from fitted weights"""
        df = self.df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])

        control_wide = (
            df[~df[self.unit_col].isin(self.treated_units)]
            .pivot(index=self.time_col, columns=self.unit_col,
                   values=self.metric_col)
            .sort_index()
        )

        y_ctrl = control_wide.to_numpy()
        synth_series = y_ctrl @ self.weights_

        treated_series = (
            df[df[self.unit_col].isin(self.treated_units)]
            .groupby(self.time_col)[self.metric_col]
            .mean()
            .reindex(control_wide.index)
        )

        results = pd.DataFrame({
            "time": control_wide.index,
            "observed": treated_series.values,
            "counterfactual": synth_series
        })
        self.results_ = results

    def estimate(self):
        self.fit() # generates weights
        self.predict() # generates self.results_ used in .summary() and .plot()

    def weight_table(self, round_to: int = 2):
        df_weights = pd.DataFrame({
            "unit": self.ctrl_units_,
            "weight": self.weights_
        })
        df_weights = df_weights[df_weights.round(round_to)["weight"] > 0]
        df_weights = df_weights.sort_values("weight", ascending=False).reset_index(drop=True)
        return df_weights      

    def summary(self, ci_level=0.95):

        df = self.results_.copy()
        post = df[df["time"] >= self.treatment_start].copy()

        post["diff"] = post["observed"] - post["counterfactual"]
        att = post["diff"].mean()

        ci_lower = np.nan
        ci_upper = np.nan

        self.summary_df_ = pd.DataFrame({
            "metric": [self.metric_col],
            "pre_R2": [self.pre_r2_],
            "ATT": [att],
            f"{int(ci_level*100)}%_lower": [ci_lower],
            f"{int(ci_level*100)}%_upper": [ci_upper]
        })

        return self.summary_df_

    def plot(self):
        df = self.results_.copy()
        #plt.figure(figsize=(8, 5))
        plt.plot(df["time"], df["observed"], color="black", label="Observed")
        plt.plot(df["time"], df["counterfactual"], color="blue", label="Counterfactual")
        plt.axvline(self.treatment_start, color="gray", linestyle="--")
        plt.legend()
        plt.xlabel("Time [Week]")
        plt.ylabel(self.metric_col.capitalize())
        plt.tight_layout()
        plt.show()


#=======================================
# interactive fixed effects estimator
# attempt to replicate the Xu et al (2017) method
# estimates Yit =Λi Ft +τ Dit +εit,
# need to add inference methods
# mostly just used this to sanity check factor simulator 
#=======================================
class IFE:
    def __init__(
        self,
        df,
        metric_col,
        unit_col,
        time_col,
        treated_units,
        treatment_start,
        K_list,
        CV=True,
        tol=1e-6,
        max_iter=100,
        cv_fraction=0.1,
        random_state=42,
        verbose=True,
    ):
        self.df = df.copy()
        self.metric_col = metric_col
        self.unit_col = unit_col
        self.time_col = time_col
        self.treated_units = [str(u) for u in treated_units]
        self.treatment_start = pd.to_datetime(treatment_start)
        self.K_list = K_list
        self.CV = CV
        self.tol = tol
        self.max_iter = max_iter
        self.cv_fraction = cv_fraction
        self.random_state = random_state
        self.verbose = verbose

        self.results_ = None
        self.Lambda_ = None
        self.F_ = None
        self.tau_ = None
        self.K_opt_ = None
        self.cv_errors_ = None
        self.pre_r2_ = None

   
    def _low_rank_completion(self, Y, mask, K):
        Y_hat = Y.copy()
        # initialize missing entries
        col_means = np.nanmean(Y_hat, axis=0)
        inds = np.where(~mask)
        Y_hat[inds] = np.take(col_means, inds[1])

        for _ in range(self.max_iter):
            U, svals, Vt = np.linalg.svd(np.nan_to_num(Y_hat), full_matrices=False)
            Lambda_hat = U[:, :K] * np.sqrt(svals[:K])
            F_hat = (Vt[:K, :].T) * np.sqrt(svals[:K])
            Y_new = Lambda_hat @ F_hat.T
            diff = np.nanmean((Y_new[mask] - Y_hat[mask]) ** 2)
            Y_hat[~mask] = Y_new[~mask]
            Y_hat[mask] = Y[mask]
            if diff < self.tol:
                break
        return Lambda_hat, F_hat, Y_hat

    def _estimate_tau(self, Y, D, mask, K):
        tau = 0.0
        for _ in range(self.max_iter):
            Y_resid = Y - tau * D
            Y_resid[~mask] = np.nan
            Lambda_hat, F_hat, _ = self._low_rank_completion(Y_resid, mask, K)
            Y_tilde = Y - Lambda_hat @ F_hat.T
            numerator = np.nansum(D * Y_tilde)
            denominator = np.nansum(D * D)
            tau_new = numerator / denominator
            if abs(tau_new - tau) < self.tol:
                break
            tau = tau_new
        return tau, Lambda_hat, F_hat

    def _estimate_tau(self, Y, D, mask, K):
        tau = 0.0
        for _ in range(self.max_iter):
            # estimate factors on observed cells only
            Y_resid = Y - tau * D
            Y_resid[~mask] = np.nan
            Λ, F, _ = self._low_rank_completion(Y_resid, mask, K)
    
            # now update tau using all cells
            Y_tilde = Y - Λ @ F.T
            num = np.nansum(D * Y_tilde)
            den = np.nansum(D * D)
            tau_new = num / den
            if abs(tau_new - tau) < self.tol:
                break
            tau = tau_new
        return tau, Λ, F        

    def _cv_select_K(self, Y, D, mask):
        rng = np.random.default_rng(self.random_state)
        pre_mask = (D == 0) & mask
        pre_idx = np.argwhere(pre_mask)
        n_mask = int(self.cv_fraction * len(pre_idx))
        mask_idx = rng.choice(len(pre_idx), n_mask, replace=False)
        mask_coords = pre_idx[mask_idx]

        cv_errors = {}
        for K_try in self.K_list:
            Y_train = Y.copy()
            train_mask = mask.copy()
            train_mask[tuple(mask_coords.T)] = False

            tau_hat, Lambda_hat, F_hat = self._estimate_tau(Y_train, D, train_mask, K_try)
            Y_pred = Lambda_hat @ F_hat.T + tau_hat * D
            residuals = (Y[tuple(mask_coords.T)] - Y_pred[tuple(mask_coords.T)])
            mse = np.nanmean(residuals**2)
            cv_errors[K_try] = mse
            if self.verbose:
                print(f"K={K_try}: CV MSE={mse:.4f}")

        K_opt = min(cv_errors, key=cv_errors.get)
        if self.verbose:
            print(f"Optimal K selected: {K_opt}")
        return K_opt, cv_errors

    def fit(self):
        df = self.df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df["treated_flag"] = df[self.unit_col].isin(self.treated_units)
        df["post_flag"] = df[self.time_col] >= self.treatment_start
        df["D"] = df["treated_flag"] & df["post_flag"]

        Y = df.pivot(index=self.unit_col, columns=self.time_col, values=self.metric_col).to_numpy()
        D = df.pivot(index=self.unit_col, columns=self.time_col, values="D").astype(float).to_numpy()

        mask = np.ones_like(Y, dtype=bool)
        unit_order = df[self.unit_col].unique()
        time_vals = df[self.time_col].sort_values().unique()
        treated_idx = [i for i, u in enumerate(unit_order) if str(u) in self.treated_units]
        post_idx = np.where(time_vals >= self.treatment_start)[0]
        mask[np.ix_(treated_idx, post_idx)] = False

        if self.CV:
            self.K_opt_, self.cv_errors_ = self._cv_select_K(Y, D, mask)
            K_use = self.K_opt_
        else:
            K_use = self.K_list[0]
            if self.verbose:
                print(f"CV disabled. Using first K in list: {K_use}")

        # estimate tau and factors on control + pre data
        tau_hat, Lambda_ctrl, F_hat = self._estimate_tau(Y, D, mask, K_use)

        #project treated unit loadings using pre-period only
        Lambda_hat = Lambda_ctrl.copy()
        pre_period = np.where(time_vals < self.treatment_start)[0]
        for i in treated_idx:
            F_pre = F_hat[pre_period, :]
            Y_pre = Y[i, pre_period]
            Lambda_hat[i, :] = np.linalg.lstsq(F_pre, Y_pre, rcond=None)[0]

        self.tau_, self.Lambda_, self.F_ = tau_hat, Lambda_hat, F_hat

        treated_idx = [i for i, u in enumerate(unit_order) if str(u) in self.treated_units]
        pre_period = np.where(time_vals < self.treatment_start)[0]
        
        Y_pred = Lambda_hat @ F_hat.T + tau_hat * D
        
        Y_obs_pre = Y[np.ix_(treated_idx, pre_period)]
        Y_pred_pre = Y_pred[np.ix_(treated_idx, pre_period)]        
        ssr = np.nansum((Y_obs_pre - Y_pred_pre) ** 2)
        sst = np.nansum((Y_obs_pre - np.nanmean(Y_obs_pre)) ** 2)
        self.pre_r2_ = 1 - ssr / sst if sst > 0 else np.nan
        if self.verbose:
            print(f"τ̂ = {tau_hat:.4f}, pre-R² = {self.pre_r2_:.3f}")

        self._Y = Y
        self._D = D
        self._mask = mask
        self._time_vals = time_vals
        self._unit_order = unit_order
        self._treated_idx = treated_idx

    def predict(self):
        Y_pred = self.Lambda_ @ self.F_.T + self.tau_ * self._D
        D_cf = self._D.copy()
        D_cf[self._treated_idx, :] = 0
        Y_cf = self.Lambda_ @ self.F_.T + self.tau_ * D_cf

        Y_obs = self._Y[self._treated_idx, :].mean(axis=0)
        Y_cf_treated = Y_cf[self._treated_idx, :].mean(axis=0)

        results = pd.DataFrame({
            "time": self._time_vals,
            "observed": Y_obs,
            "counterfactual": Y_cf_treated
        })
        self.results_ = results

    def estimate(self):
        self.fit()
        self.predict()

    def summary(self, ci_level=0.95):
        df = self.results_.copy()
        post = df[df["time"] >= self.treatment_start].copy()
        post["diff"] = post["observed"] - post["counterfactual"]
        att = post["diff"].mean()

        summary_df = pd.DataFrame({
            "metric": [self.metric_col],
            "pre_R2": [self.pre_r2_],
            "ATT": [att],
            "tau_hat": [self.tau_],
            "K_used": [self.K_opt_ if self.CV else self.K_list[0]]
        })
        self.summary_df_ = summary_df
        return summary_df

    def plot(self):
        df = self.results_.copy()
        plt.plot(df["time"], df["observed"], color="black", label="Observed")
        plt.plot(df["time"], df["counterfactual"], color="blue", label="Counterfactual")
        plt.axvline(self.treatment_start, color="gray", linestyle="--")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(self.metric_col.capitalize())
        plt.tight_layout()
        plt.show()


class FDID(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        df: pd.DataFrame,
        unit_col: str,
        time_col: str,
        metric_col: str,
        treated_units: list[str],
        treatment_start: str,
        alpha: float = 0.05,
        control_units: list[str] | None = None,
    ) -> None:
        """
        Forward Difference-in-Differences (FDID) Estimator
        based on Li (2024)
        https://pubsonline.informs.org/doi/abs/10.1287/mksc.2022.0212?journalCode=mksc

        Parameters:
        - df (pd.DataFrame): Input data.
        - metric_col (str): Column name for the metric.
        - unit_col (str): Column name for the unit.
        - time_col (str): Column name for time.
        - treated_units (list): List of treated units.
        - treatment_start (str): Start of treatment period.
        - alpha (float): Significance level for confidence intervals.
        """
        self.df = df.copy()
        self.unit_col = unit_col
        self.time_col = time_col
        self.metric_col = metric_col
        self.treated_units = treated_units
        self.control_units = control_units
        self.treatment_start = str(treatment_start)
        self.alpha = alpha

    def process_input_data(self) -> None:
        """
        Process the user-defined data.
        Removes imbalanced units, trims outliers if requested, and creates `treated_unit` and
        `treatment_period` columns
        """
        if not is_string_dtype(self.df[self.unit_col]):
            self.df[self.unit_col] = self.df[self.unit_col].astype(str)
        if not is_string_dtype(self.df[self.time_col]):
            self.df[self.time_col] = self.df[self.time_col].astype(str)

    def prepare_estimation_data(self) -> None:
        """
        Set up X and Y for fitting and prediction
        """
        df = self.df.copy()

        df[self.unit_col] = df[self.unit_col].astype(str)
        df[self.time_col] = df[self.time_col].astype(str)

        df["treatment"] = np.where(
            (df[self.unit_col].isin(self.treated_units)) &
            (df[self.time_col] > self.treatment_start), True, False,
        )

        treated_df = df[df[self.unit_col].isin(self.treated_units)]

        if len(self.treated_units) > 1:
            treated_df = treated_df.groupby([self.time_col, "treatment"], as_index=False)[self.metric_col].mean()

        control_df = df[~df[self.unit_col].isin(self.treated_units)]
        self.X = control_df.pivot(index=self.time_col, columns=self.unit_col, values=self.metric_col)
        self.X = self.X.loc[treated_df[self.time_col]]

        self.y = pd.Series(treated_df[self.metric_col].values, index=treated_df[self.time_col])

        self.pre_treatment = self.X.index[self.X.index <= self.treatment_start]
        self.post_treatment = self.X.index[self.X.index > self.treatment_start]

        self.X_pre, self.y_pre = self.X.loc[self.pre_treatment], self.y.loc[self.pre_treatment]

    def _select_controls(self, X_pre: np.ndarray, y_pre: np.ndarray) -> list[str]:
        """
        Selects control units using greedy forward selection

        Parameters:
        - X_pre: Pre-treatment control data (time × control units)
        - y_pre: Pre-treatment treated outcome

        Returns:
        - selected_controls: list of names of selected control units,(eg regions)
        """
        num_units = X_pre.shape[1]
        potential_controls = list(range(num_units))
        unit_names = list(self.X.columns)

        # compute initial R² for each control unit separately
        initial_r2 = np.array([self._get_r_squared(X_pre[:, i], y_pre) for i in potential_controls])

        # start with best initial control (highest R²)
        first_control = np.argmax(initial_r2)
        control_pool = [first_control]
        incumbent_r2 = initial_r2[first_control]

        remaining_controls = set(potential_controls) - {first_control}

        R2_values = [incumbent_r2]
        control_history = [control_pool.copy()]

        # forward through remaining controls
        for _ in range(1, num_units):
            R2 = []

            for candidate in remaining_controls:
                new_pool = control_pool + [candidate]
                r2 = self._get_r_squared(X_pre[:, new_pool].mean(axis=1), y_pre)
                R2.append(r2)

            best_new_control = list(remaining_controls)[np.argmax(R2)]
            best_r2 = max(R2)

            control_pool.append(best_new_control)
            remaining_controls.remove(best_new_control)

            R2_values.append(best_r2)
            control_history.append(control_pool.copy())

        best_iteration = np.argmax(R2_values)
        best_controls = control_history[best_iteration]
        selected_controls = [unit_names[i] for i in best_controls]

        return selected_controls

    def fit(self) -> None:
        """
        Estimate the counterfactual. Only fits one parameter, the intercept.
        """
        if self.control_units is None:
            self.did_controls = self._select_controls(self.X_pre.values, self.y_pre.values)
        else:
            self.did_controls = self.controls
        self.did_controls = self._select_controls(self.X_pre.values, self.y_pre.values)
        self.alpha_U = np.mean(self.y_pre - np.mean(self.X_pre.loc[:, self.did_controls], axis=1))
        self.pre_r_squared = self._get_r_squared(np.mean(self.X_pre.loc[:, self.did_controls], axis=1), self.y_pre)
        self.pre_rmse = self._get_rmse(np.mean(self.X_pre.loc[:, self.did_controls], axis=1), self.y_pre)

    def predict(self, inference: str = 'iid') -> None:
        """
        Predict the counterfactual and estimate confidence intervals.
        """

        self.inference_method = inference

        self.yhat = self.alpha_U + np.mean(self.X.loc[:, self.did_controls], axis=1)
        self.yhat = pd.Series(self.yhat, index=self.X.index)
        self.u_pre = self.y[self.pre_treatment] - self.yhat[self.pre_treatment]
        self.u_post = self.y[self.post_treatment] - self.yhat[self.post_treatment]
        if inference == 'newey_west':
            self.se = self._se_newey_west()
        else:
            self.se = self._se_iid()
        z_scores = scipy.stats.norm.ppf([self.alpha / 2, 1 - self.alpha / 2])
        lower = self.yhat + self.se * z_scores[0]
        upper = self.yhat + self.se * z_scores[1]
        self.results = pd.DataFrame({
            self.time_col: self.y.index,
            'metric': self.metric_col,
            'observed': self.y.values,
            'counterfactual': self.yhat,
            'lower': lower,
            'upper': upper,
        }).reset_index(drop=True)

    def estimate(self, controls: list[str] | None = None, inference: str = 'iid') -> None:
        """
        Main user-facing method. Process, prepare, fit and predict.
        Updates self.results which can then be plotted
        """
        self.process_input_data()
        self.prepare_estimation_data()
        self.controls = controls
        self.fit()
        self.predict(inference=inference)

    def summary(self) -> pd.DataFrame:
        """
        Publish a summary table of the estimator + intervals
        """
        self.att = np.mean(self.y[self.post_treatment] - self.yhat[self.post_treatment])
        self.rel_att = (
            np.sum(self.y[self.post_treatment] - self.yhat[self.post_treatment])
            / np.sum(self.yhat[self.post_treatment])
        )
        z_value = self.att / self.se
        self.p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_value)))
        interval_width = int(100 * np.round(1 - self.alpha, 2))
        self.summary_df = pd.DataFrame({
            "estimator": ["FDID"],
            "interval_width (%)": [interval_width],
            "inference_method": [self.inference_method],
            "pre-treatment R2": [self.pre_r_squared],
            "pre-treatment RMSE": [self.pre_rmse],
            "att": [self.att],
            "rel_att": [self.rel_att * 100.0],
            "se": [self.se],
            "lower": [self.att - scipy.stats.norm.ppf(1-self.alpha/2) * self.se],
            "upper": [self.att + scipy.stats.norm.ppf(1-self.alpha/2) * self.se],
            "p_value": [self.p_value],
        })
        return self.summary_df

    def plot(self):
        df = self.results.copy()
    
        fig, ax = plt.subplots(figsize=(8, 5))
    
        ax.plot(df[self.time_col], df["observed"], color="black", label="Observed")
        ax.plot(df[self.time_col], df["counterfactual"], color="blue", label="Counterfactual")
    
        post_mask = df[self.time_col] >= self.treatment_start
        ax.fill_between(
            df.loc[post_mask, self.time_col],
            df.loc[post_mask, "lower"],
            df.loc[post_mask, "upper"],
            color="blue",
            alpha=0.2,
            label="95% CI"
        )
    
        ax.axvline(self.treatment_start, color="gray", linestyle="--")
    
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    
        ax.set_xlabel("Time [Week]")
        ax.set_ylabel(self.metric_col.capitalize())
        ax.legend()
        plt.tight_layout()
        plt.show()      

    def _get_r_squared(self, x: np.ndarray, y: np.ndarray) -> np.float:
        yhat = np.mean(y - x) + x
        return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    def _get_rmse(self, x: np.ndarray, y: np.ndarray) -> float:
        yhat = np.mean(y - x) + x
        rmse = np.sqrt(np.mean((y - yhat) ** 2))
        return rmse        

    def _se_iid(self) -> np.float:
        T1 = len(self.pre_treatment)
        var_u_pre = np.sum(self.u_pre**2) / T1
        se = np.sqrt(var_u_pre)
        return se

    def _se_newey_west(self) -> np.float:
        T1 = len(self.pre_treatment)
        T2 = len(self.post_treatment)
        u_pre_array = np.asarray(self.u_pre).reshape(-1, 1)
        max_lag = int(T1 ** (1/4))
        v = sm.stats.sandwich_covariance.S_hac_simple(u_pre_array, nlags=max_lag)[0, 0]
        se = np.sqrt(v / T2)
        return se

    def dickey_fuller_test(self) -> None:
        """
        Run a Dickey-Fuller test for stationarity of pre-treatment
        and fit an ARIMA(1,0,0) = AR(1) process on pre-treatment
        """
        adf_result = adfuller(self.u_pre)

        ar_model = ARIMA(self.u_pre, order=(1, 0, 0)).fit()
        rho_est = ar_model.params[1]
        rho_se = ar_model.bse[1]
        rho_pval = ar_model.pvalues[1]

        equation_text = "AR(1) Model:  y_t = ρ y_{t-1} + ε_t"
        null_hypothesis = "H₀: ρ = 1 (Unit root, non-stationary)"
        alt_hypothesis = "H₁: ρ < 1 (Stationary)"
        test_info = f"""
        {equation_text}

        {null_hypothesis}
        {alt_hypothesis}
        """

        headers = ["Test Statistic", "p-value", "Lags Used", "Observations"]
        data = [[f"{adf_result[0]:.4f}", f"{adf_result[1]:.4f}", adf_result[2], adf_result[3]]]

        ar_headers = ["Estimated ρ", "standard error", "p-value"]
        ar_data = [[f"{rho_est:.4f}", f"{rho_se:.4f}",  f"{rho_pval:.4f}"]]

        adf_table = SimpleTable(data, headers, title="Augmented Dickey-Fuller Test Results")
        ar_table = SimpleTable(ar_data, ar_headers, title="Estimated AR(1) Coefficient")

        print(adf_table)
        print(test_info)
        print(ar_table)        
        

    def hte(self, verbose) -> None:
        """
        Estimate heterogeneous treatment effects by fitting separate FDID models
        for each treated unit, excluding all other treated units from the control pool.
        """
        if not hasattr(self, 'results'):
            raise ValueError("`estimate()` must be run before `hte()`")

        summaries = []
        pbar = tqdm(self.treated_units, desc="Estimating HTEs", disable=not verbose)
        for treated_unit in pbar:
            pbar.set_description(f"Estimating ATT for treated unit: {treated_unit}")

            # exclude all treated units except the current one
            df_filtered = self.df[
                ~self.df[self.unit_col].isin(self.treated_units) | (self.df[self.unit_col] == treated_unit)
            ]

            # temporary estimator with the current treated unit
            temp_estimator = self.__class__(
                df=df_filtered,
                metric_col=self.metric_col,
                unit_col=self.unit_col,
                time_col=self.time_col,
                treated_units=[treated_unit],
                treatment_start=self.treatment_start,
            )

            # run the estimation pipeline
            temp_estimator.estimate()
            summary_df = temp_estimator.summary()
            summary_df['treated_unit'] = treated_unit
            summaries.append(summary_df)

        hte_results = pd.concat(summaries, ignore_index=True)
        # move treated_unit col to the front
        cols = ['treated_unit'] + [col for col in hte_results.columns if col != 'treated_unit']
        self.hte_results = hte_results[cols]        



class ForwardDesign:
    """
    A class to iteratively select treated units using a forward greedy algorithm
    to maximize pre-treatment R² in the Forward Diff-in-Diff estimator
    """

    def __init__(
        self,
        df: pd.DataFrame,
        unit_col: str,
        time_col: str,
        metric_col: str,
        treatment_start: str,
    ) -> None:
        """
        Initialize the ForwardDesign class

        Parameters:
            df (pd.DataFrame): Panel data containing the outcome and unit (e.g., rides x regions)
            unit_col (str): Column name identifying the units (e.g., regions)
            time_col (str): Column name for the time variable
            metric_col (str): Column name for the outcome variable
            treatment_start (str): Start date of the treatment period
        """
        self.df = df
        self.unit_col = unit_col
        self.time_col = time_col
        self.metric_col = metric_col
        self.treatment_start = treatment_start

    def solve(self, tolerance: float = 0.001, verbose: bool = True) -> tuple[list[str], float]:
        """
        Selects treated units iteratively to maximize pre-treatment R²

        Parameters:
            tolerance (float, optional): Minimum R² improvement required to
            continue adding treated units.

        Returns:
            tuple: A list of selected treated units and the final
            pre-treatment R² value
        """
        self.verbose = verbose
        self.tolerance = tolerance
        units = self.df[self.unit_col].unique()
        initial_r2 = []

        for u in tqdm(units, desc="Screening first treated unit", disable=not verbose):
            fdid = FDID(
                df=self.df,
                metric_col=self.metric_col,
                unit_col=self.unit_col,
                time_col=self.time_col,
                treated_units=[u],
                treatment_start=self.treatment_start,
            )
            fdid.process_input_data()
            fdid.prepare_estimation_data()
            fdid.fit()
            initial_r2.append(fdid.pre_r_squared)

        initial_r2 = np.nan_to_num(initial_r2, nan=-np.inf)
        first_treated = np.argmax(initial_r2)
        treated_pool = [units[first_treated]]
        incumbent_r2 = initial_r2[first_treated]
        remaining_units = set(units) - {treated_pool[0]}

        while True:
            R2 = []
            candidates = list(remaining_units)

            for candidate in tqdm(candidates, desc="Adding more treated units   ", disable=not verbose):
                new_pool = treated_pool + [candidate]
                fdid = FDID(
                    df=self.df,
                    metric_col=self.metric_col,
                    unit_col=self.unit_col,
                    time_col=self.time_col,
                    treated_units=new_pool,
                    treatment_start=self.treatment_start,
                )
                fdid.process_input_data()
                fdid.prepare_estimation_data()
                fdid.fit()
                R2.append(fdid.pre_r_squared)

            best_new_treated = candidates[np.argmax(R2)]
            best_r2 = max(R2)

            if best_r2 < incumbent_r2 or abs(best_r2 - incumbent_r2) < self.tolerance:
                break

            treated_pool.append(best_new_treated)
            incumbent_r2 = best_r2
            remaining_units.remove(best_new_treated)

        self.treated_pool = treated_pool
        self.optimal_r2 = incumbent_r2

        self.fdid_optimal = FDID(
            df=self.df,
            metric_col=self.metric_col,
            unit_col=self.unit_col,
            time_col=self.time_col,
            treated_units=self.treated_pool,
            treatment_start=self.treatment_start,
        )

        self.fdid_optimal.estimate()

        self.control_pool = self.fdid_optimal.did_controls

        self.design_df = self.df[self.df[self.unit_col].isin(self.treated_pool + self.control_pool)]

        if self.verbose:
            return self.treated_pool, self.control_pool, self.optimal_r2

    def run_aa(self) -> None:

        df_aa = self.df[self.df[self.unit_col].isin(self.control_pool + self.treated_pool)]

        times = df_aa[self.time_col].unique()

        post_treatment_length = len(times[times > np.datetime64(self.treatment_start)])

        # create a bunch of placebo treatment times
        # might need a more intelligent method or some defensive checks
        self.aa_times = times[(len(times)//3):(len(times)-post_treatment_length)]

        self.aa_r2 = []

        self.aa_pvalue = []

        self.aa_tau = []

        frequency = infer_time_freq(times)

        for i in tqdm(range(len(self.aa_times))):
            cutoff = self.aa_times[i] + np.timedelta64(post_treatment_length, frequency)
            df_aa_test = df_aa[df_aa[self.time_col] <= cutoff]
            fdid = FDID(
                df=df_aa_test,
                metric_col=self.metric_col,
                unit_col=self.unit_col,
                time_col=self.time_col,
                treated_units=self.treated_pool,
                treatment_start=self.aa_times[i],
            )
            fdid.estimate()
            fdid.summary()
            self.aa_tau.append(fdid.att)
            self.aa_r2.append(fdid.pre_r_squared)
            self.aa_pvalue.append(fdid.p_value)


    def plot_aa(self, result_type: str = 'fpr') -> None:
        """
        Plot the distribution of:
            - placebo treatment effects (ATT)
            - placebo p-values (false-positive rate)
        Defaults to FPR
        """
        if result_type == 'att':
            plt.hist(self.aa_tau, bins=50, alpha=0.75)
            plt.axvline(x=np.mean(self.aa_tau), color='black', linewidth=3)
            plt.xlabel("Estimated ATT")
            plt.ylabel("Count")
            plt.title("AA Tests with optimal treated/control config", fontsize=14)
            plt.tight_layout()
            plt.show()

        elif result_type == 'fpr':
            fpr = 100.0 * np.round(np.mean(np.array(self.aa_pvalue) < 0.05), 3)
            plt.hist(self.aa_pvalue, bins=50, alpha=0.75)
            plt.axvline(x=0.05, color='black', linewidth=3)
            plt.xlabel("p-value")
            plt.ylabel("Count")
            plt.text(0.065, plt.gca().get_ylim()[1] * 0.9, "average p < 0.05", fontsize=9, va='center')
            plt.title("AA Tests with optimal treated/control config\n", fontsize=14)
            plt.suptitle(f"$\\text{{FPR}} \\approx {fpr:.1f}\\%$\n", y=0.91, fontsize=10)
            plt.tight_layout()
            plt.show()

        else:
            raise ValueError("type must be either 'att' or 'fpr'")

    def simulate_effect(
        self,
        effect: float,
        treatment_start: str = None,
    ) -> FDID:
        """
        Simulate a treatment effect on the design_df using the optimal treated/control configuration
        and re-estimate it with FDID
        """
        df_sim = self.design_df.copy()
    
        #df_sim[self.time_col] = pd.to_datetime(df_sim[self.time_col])
    
        # determine post-treatment window
        if treatment_start is None:
            treatment_start = self.treatment_start
        
        post_mask = df_sim[self.time_col] >= treatment_start
        
        treated_mask = df_sim[self.unit_col].isin(self.treated_pool)
    
        # apply treatment effect (fixed to relative for now)
        mask = post_mask & treated_mask
        df_sim.loc[mask, self.metric_col] *= (1 + effect)
        df_sim[self.time_col] = df_sim[self.time_col].dt.strftime("%Y-%m-%d")
    
        # estimate
        fdid = FDID(
            df=df_sim,
            unit_col=self.unit_col,
            time_col=self.time_col,
            metric_col=self.metric_col,
            treated_units=self.treated_pool, # plug in the optimal treated
            control_units=self.control_pool, # plug in the optimal controls
            treatment_start=treatment_start,
        )
    
        fdid.estimate(inference='iid')

        return fdid


    def sample_times(self, times: list[str], n_treatment_times: int, num_simulated_dates: int = 10):
        """
        Randomly sample treatment start and end times for simulation.
        """
        n_time = len(times)
        n_ctrl_time = n_time - n_treatment_times
        treatment_start_idx = np.random.randint(
            low=n_ctrl_time // 2, high=n_ctrl_time, size=num_simulated_dates
        )
        treatment_start_times = [times[i] for i in treatment_start_idx]
        treatment_end_times = [times[i + n_treatment_times - 1] for i in treatment_start_idx]
        return treatment_start_times, treatment_end_times

    
    def check_power(
        self,
        effect_sizes: list[float],
        vary_dates: bool = False,
        n_simulated_dates: int = 20,
        n_treatment_times: int = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Simulate experiments across multiple effect sizes and estimate power.
        Uses simulate_effect() with optimal treated/control configuration.
    
        Parameters
        ----------
        effect_sizes : list[float]
            List of treatment effects to simulate (e.g. [0.0, 0.05, 0.10, 0.15]).
        vary_dates : bool, optional
            Whether to vary treatment start and end dates across simulations.
        n_simulated_dates : int, optional
            Number of simulated treatment start times to test (if vary_dates=True).
        n_treatment_times : int, optional
            Length of treatment period in time units.
    
        Returns
        -------
        pd.DataFrame
            A DataFrame with treatment start/end, effect, estimate, CI bounds, and rejection indicator.
        """
        dfs = []
    
        if vary_dates:
            times = sorted(self.df[self.time_col].unique())
            treatment_start_times, treatment_end_times = self.sample_times(
                times, n_treatment_times, n_simulated_dates
            )
        else:
            treatment_start_times = [self.treatment_start]
            treatment_end_times = [self.df[self.time_col].max()]
    
        # loop over effect sizes and treatment dates
        for eff in tqdm(effect_sizes, desc="Simulating effect sizes", disable=not verbose):
            for start, end in zip(treatment_start_times, treatment_end_times):
                # Temporarily override treatment_start for this simulation
                #self.treatment_start = start # this is the bug where I override the original self.treatment start
                fdid = self.simulate_effect(effect=eff,treatment_start=start) # have this take an optional treatmnet_start? 
                summary = fdid.summary()
    
                # reject null (or not)
                lower, upper = summary["lower"].iloc[0], summary["upper"].iloc[0]
                reject = int(lower * upper > 0)
    
                dfs.append({
                    "treatment_start": start,
                    "treatment_end": end,
                    "effect": eff,
                    "estimate": summary["rel_att"].iloc[0],
                    "lower": lower,
                    "upper": upper,
                    "reject": reject,
                    "pre_r_squared": fdid.pre_r_squared,
                    "pre_rmse": fdid.pre_rmse
                })
    
        self.power_df = pd.DataFrame(dfs)
        return self.power_df



    def plot_power(self, smooth=True, n_points=300, target_power=0.8):
        """
        Plot an interpolated power curve with smoother interpolation and
        display both positive and negative MDEs (Minimum Detectable Effects).
    
        Parameters
        ----------
        smooth : bool, optional
            If True, interpolate smoothly between simulated points.
        n_points : int, optional
            Number of points for fine interpolation grid.
        target_power : float, optional
            Desired power threshold (default 0.8).
        """
        power_curve = (
            self.power_df.groupby('effect', as_index=False)['reject'].mean()
            .sort_values('effect')
            .reset_index(drop=True)
        )
    
        effects = power_curve['effect'].values
        power_vals = power_curve['reject'].values
    
        if smooth and len(effects) > 2:
            f_interp = PchipInterpolator(effects, power_vals)
            x_fine = np.linspace(min(effects), max(effects), n_points)
            y_fine = np.clip(f_interp(x_fine), 0, 1)
        else:
            x_fine, y_fine = effects, power_vals
    
        pos_mask = x_fine >= 0
        neg_mask = x_fine <= 0
    
        mde_pos = np.nan
        mde_neg = np.nan
    
        if np.any(y_fine[pos_mask] >= target_power):
            idx_pos = np.where(y_fine[pos_mask] >= target_power)[0][0]
            mde_pos = x_fine[pos_mask][idx_pos]
    
        if np.any(y_fine[neg_mask] >= target_power):
            idx_neg = np.where(y_fine[neg_mask][::-1] >= target_power)[0][0]
            mde_neg = x_fine[neg_mask][::-1][idx_neg]
    
        plt.figure(figsize=(7, 5))
        plt.plot(x_fine, y_fine, '-', lw=2, label='Interpolated', color='C0')
        #plt.scatter(effects, power_vals, color='C0', s=35, label='Simulated (raw)', alpha=0.7)
        plt.axhline(target_power, color='k', ls='--', label=f'{int(target_power*100)}% power')
    
        if not np.isnan(mde_pos):
            plt.axvline(mde_pos, color='r', ls=':', lw=2)
            plt.text(
                mde_pos, target_power + 0.05,
                f"MDE+ ≈ {mde_pos*100:.1f}%", color='r',
                ha='left', fontsize=10, weight='bold'
            )
        if not np.isnan(mde_neg):
            plt.axvline(mde_neg, color='r', ls=':', lw=2)
            plt.text(
                mde_neg, target_power + 0.05,
                f"MDE− ≈ {mde_neg*100:.1f}%", color='r',
                ha='right', fontsize=10, weight='bold'
            )
    
        plt.xlabel("Effect Size")
        plt.ylabel("Power")
        plt.title("Interpolated Power Curve", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
        plt.tight_layout()
        plt.show()

    
    def check_power_random(
        self,
        effect_sizes: list[float],
        n_treated: int | None = None,
        n_controls: int | None = None,
        n_reps: int = 10,
        treatment_start_times: list[pd.Timestamp] | None = None,
        n_treatment_times: int = 4,
        n_jobs: int = -1,
        verbose: bool = True,
    ):
        """
        Compare power for random treated/control configurations
        using the same treatment windows as the forward design
    
        Parameters
        ----------
        effect_sizes : list[float]
            List of treatment effects to simulate.
        n_treated : int, optional
            Number of treated units to randomly sample (defaults to len(self.treated_pool))
        n_controls : int, optional
            Number of control units to randomly sample (defaults to len(self.control_pool))
        n_reps : int
            Number of random draws per effect size.
        treatment_start_times : list[pd.Timestamp], optional
            List of treatment start times (e.g., from self.sample_times())
            If None, uses [self.treatment_start]
        n_treatment_times : int
            Length of treatment period in time units (e.g., days or weeks)
        """
        units = self.df[self.unit_col].unique()
        n_treated = n_treated or len(self.treated_pool)
        n_controls = n_controls or len(self.control_pool)
        treatment_start_times = treatment_start_times or [pd.to_datetime(self.treatment_start)]
        
        df_base = self.df.copy()
        df_base[self.time_col] = pd.to_datetime(df_base[self.time_col]).dt.to_period("W").dt.start_time
    
        def simulate_once(eff, t_start, rep):
            t_end = t_start + pd.to_timedelta(n_treatment_times, unit="W")
    
            treated_rand = np.random.choice(units, n_treated, replace=False)
            remaining = np.setdiff1d(units, treated_rand)
            control_rand = np.random.choice(remaining, n_controls, replace=False)
    
            df_sim = df_base[df_base[self.time_col] <= t_end].copy()
            mask = (
                df_sim[self.unit_col].isin(treated_rand)
                & (df_sim[self.time_col] >= t_start)
                & (df_sim[self.time_col] < t_end)
            )
    
            df_sim.loc[mask, self.metric_col] *= (1 + eff)
            df_sim[self.time_col] = df_sim[self.time_col].dt.strftime("%Y-%m-%d")
    
            fdid = FDID(
                df=df_sim,
                unit_col=self.unit_col,
                time_col=self.time_col,
                metric_col=self.metric_col,
                treated_units=list(treated_rand),
                control_units=list(control_rand),
                treatment_start=t_start.strftime("%Y-%m-%d"),
            )
            fdid.estimate(inference="iid")
            summ = fdid.summary()
    
            lower, upper = summ["lower"].iloc[0], summ["upper"].iloc[0]
            reject = int((lower > 0) | (upper < 0))  # two-sided
    
            return {
                "design": "random",
                "treatment_start": t_start,
                "treatment_end": t_end,
                "effect": eff,
                "estimate": summ["rel_att"].iloc[0],
                "lower": lower,
                "upper": upper,
                "reject": reject,
                "pre_r_squared": fdid.pre_r_squared,
                "pre_rmse": fdid.pre_rmse,
            }
    
        tasks = [
            delayed(simulate_once)(eff, t_start, rep)
            for eff in effect_sizes
            for t_start in treatment_start_times
            for rep in range(n_reps)
        ]
    
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            tqdm(tasks, desc="Random design simulations", disable=not verbose)
        )
    
        return pd.DataFrame(results)        
