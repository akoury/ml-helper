import timeit
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
import matplotlib.pyplot as plt
from scipy.stats import variation
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from vecstack import StackingTransformer
from scipy.stats import chi2_contingency
from scipy.stats.mstats import winsorize
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import (
    cross_val_score,
    RandomizedSearchCV,
    TimeSeriesSplit,
    StratifiedKFold,
)


class Helper:
    '''
    Functions to speed up the machine learning process.

    Args:
        keys (dict): Dictionary used to set the multiple properties of the object.

    Attributes:
        KEYS (str): This is where we store all the keys.
        SEED (int): The seed used for reproducibility.
        TARGET (str): The name of the target variable.
        METRIC (str): The metric used for modeling.
        TIMESERIES (bool): Whether it is a time series or not
        SPLITS (int): The number of splits to do in cross validation
    '''
    
    def __init__(self, keys):
        self.KEYS = keys
        self.SEED = keys["SEED"]
        self.TARGET = keys["TARGET"]
        self.METRIC = keys["METRIC"]
        self.TIMESERIES = keys["TIMESERIES"]
        self.SPLITS = keys["SPLITS"]

    # Data Exploration

    def missing_data(self, df):
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
        return pd.concat([total, percent], axis=1, keys=["Total", "Percent"])

    def boxplot(self, df, exclude=[]):
        plt.figure(figsize=(12, 10))
        num = self.numericals(df, exclude)
        num = (num - num.mean()) / num.std()
        ax = sns.boxplot(data=num, orient="h")

    def coefficient_variation(self, df, threshold=0.05, exclude=[]):
        plt.figure(figsize=(8, 6))
        cols = self.numericals(df, exclude)
        variance = variation(cols)
        ax = sns.barplot(x=np.sort(variance)[::-1], y=cols.columns)

        cols = [x for x in cols.columns[np.argwhere(variance < threshold)]]
        if len(cols) > 0:
            print(str(cols) + " are invariant with a threshold of " + str(threshold))
        else:
            print("No invariant columns")
        return cols

    def correlated(self, df, threshold=0.9):
        categoric = self.categorical_correlated(df, threshold)
        numeric = self.numerical_correlated(df, threshold)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            categoric[1], cbar=True, fmt=" .2f", annot=True, cmap="viridis"
        ).set_title("Categorical Correlation", fontsize=30)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            numeric[1], cbar=True, fmt=" .2f", annot=True, cmap="viridis"
        ).set_title("Numerical Correlation", fontsize=30)

        correlated_cols = categoric[0] + numeric[0]

        if len(correlated_cols) > 0:
            print(
                "The following columns are correlated with a threshold of "
                + str(threshold)
                + ": "
                + str(correlated_cols)
            )

            if self.TARGET in correlated_cols:
                print(
                    "The target variable is correlated, consider removing its correlated counterpart"
                )
                correlated_cols.remove(self.TARGET)
        else:
            print("No correlated columns for the  " + str(threshold) + " threshold")

        return correlated_cols

    def numerical_correlated(self, df, threshold=0.9):
        corr_matrix = np.absolute(
            df.select_dtypes(include=[np.number]).corr(method="spearman")
        ).abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )
        return (
            [column for column in upper.columns if any(abs(upper[column]) > threshold)],
            corr_matrix,
        )

    def cramers_v(self, x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    def categorical_correlated(self, df, threshold=0.9):
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        corr = pd.DataFrame(index=columns, columns=columns)
        for i in range(0, len(columns)):
            for j in range(i, len(columns)):
                if i == j:
                    corr[columns[i]][columns[j]] = 1.0
                else:
                    cell = self.cramers_v(df[columns[i]], df[columns[j]])
                    corr[columns[i]][columns[j]] = cell
                    corr[columns[j]][columns[i]] = cell
        corr.fillna(value=np.nan, inplace=True)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        return (
            [column for column in upper.columns if any(abs(upper[column]) > threshold)],
            corr,
        )

    def under_represented(self, df, threshold=0.99):
        under_rep = []
        for column in df:
            counts = df[column].value_counts()
            majority_freq = counts.iloc[0]
            if (majority_freq / len(df)) > threshold:
                under_rep.append(column)

        if not under_rep:
            print("No underrepresented features")
        else:
            if self.TARGET in under_rep:
                print("The target variable is underrepresented, consider rebalancing")
                under_rep.remove(self.TARGET)
            print(str(under_rep) + " underrepresented")

        return under_rep

    def feature_importance(self, df, model, convert=False):
        X, y = self.split_x_y(df)

        if convert:
            X = self.one_hot_encode(X, self.categoricals(X))
        model.fit(X, y)
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        indices = np.argsort(importances)

        print("Feature ranking:")
        plt.figure(figsize=(16, 14))
        plt.title("Feature importances")
        plt.barh(
            range(X.shape[1]),
            importances[indices],
            color="r",
            xerr=std[indices],
            align="center",
        )
        plt.yticks(range(X.shape[1]), [list(X)[i] for i in indices])
        plt.ylim([-1, X.shape[1]])
        plt.show()

    def plot_pca_components(self, df, variance=0.9, convert=False):
        X, y = self.split_x_y(df)

        if convert:
            X = self.one_hot_encode(X, self.categoricals(X))

        pca = PCA().fit(X)

        sns.set_style("whitegrid")
        plt.figure(figsize=(9, 7))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("Number of Components")
        plt.ylabel("Variance (%)")
        plt.show()

    def target_distribution(self, df):
        plt.figure(figsize=(8, 7))
        target_count = (df[self.TARGET].value_counts() / len(df)) * 100
        target_count.plot(kind="bar", title="Target Distribution (%)")

    # Data Preparation

    def convert_to_category(self, df, cols):
        for i in cols:
            df[i] = df[i].astype("category")
        return df

    def drop_columns(self, df, cols):
        return df.drop(df[cols], axis=1)

    # Modeling

    def cross_val(self, df, model, splits=None, pipe=None, grid=None):
        if splits is None:
            splits = self.SPLITS

        X, y = self.split_x_y(df)

        if self.TIMESERIES:
            folds = TimeSeriesSplit(n_splits=splits)
        else:
            folds = StratifiedKFold(
                n_splits=splits, shuffle=True, random_state=self.SEED
            )

        if pipe:
            pipe_cv = clone(pipe)
            pipe_cv.steps.append((model["name"], model["model"]))
            model = pipe_cv

        if grid:
            model = RandomizedSearchCV(
                model,
                grid,
                scoring=self.METRIC,
                cv=folds,
                n_iter=10,
                refit=True,
                return_train_score=False,
                error_score=0.0,
                n_jobs=-1,
                random_state=self.SEED,
            )
            model.fit(X, y)
            scores = model.cv_results_["mean_test_score"]
        else:
            scores = cross_val_score(
                model, X, y, scoring=self.METRIC, cv=folds, n_jobs=-1
            )

        return scores, model

    def pipeline(
        self, df, models, pipe, all_scores=pd.DataFrame(), splits=None, note="", quiet=False
    ):
        if splits is None:
            splits = self.SPLITS

        for model in models:
            if (
                len(all_scores) == 0
                or len(
                    all_scores[
                        (all_scores["Model"] == model["name"])
                        & (all_scores["Steps"] == ", ".join(self.pipe_steps(pipe)))
                    ]
                )
                == 0
            ):
                try:
                    start = timeit.default_timer()

                    scores, cv_model = self.cross_val(
                        df.copy(), model, pipe=pipe, splits=splits
                    )

                except Exception as error:
                    cv_model = pipe
                    note = "Error: " + str(error)
                    print(note)
                    scores = np.array([0])

                all_scores = self.score(
                    model["name"],
                    scores,
                    timeit.default_timer(),
                    start,
                    cv_model,
                    note,
                    all_scores,
                )

            else:
                print(
                    str(model["name"])
                    + " already trained on those parameters, ignoring"
                )
        
        if not quiet:
            self.show_scores(all_scores)

        return all_scores

    def predict(self, df, holdout, pipe):
        X_train, y_train = self.split_x_y(df)
        pipe.fit(X_train, y_train)

        X, y = self.split_x_y(holdout)

        return y, pipe.predict(X)

    def stack_predict(self, df, holdout, pipes, amount=2):
        X, y = self.split_x_y(df)
        X_test, y_test = self.split_x_y(holdout)

        pipe = Pipeline(self.top_pipeline(pipes).steps[:-1])
        X = pipe.fit_transform(X)
        X_test = pipe.transform(X_test)

        estimators = []

        for i in range(amount):
            estimators.append((str(i), self.top_pipeline(pipes, i).steps[-1][1]))

        regression = False

        if self.METRIC in [
            "explained_variance",
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "neg_mean_squared_log_error",
            "neg_median_absolute_error",
            "r2",
        ]:
            regression = True

        stack = StackingTransformer(estimators, regression)
        stack.fit(X, y)

        S_train = stack.transform(X)
        S_test = stack.transform(X_test)

        final_estimator = estimators[0][1]
        final_estimator.fit(S_train, y)

        return final_estimator, y_test, final_estimator.predict(S_test)

    # Others

    def types(self, df, types, exclude=None):
        types = df.select_dtypes(include=types)
        excluded = [self.TARGET]
        if exclude:
            for i in exclude:
                excluded.append(i)
        cols = [col for col in types.columns if col not in excluded]
        return df[cols]

    def numericals(self, df, exclude=None):
        return self.types(df, [np.number], exclude)

    def categoricals(self, df, exclude=None):
        return self.types(df, ["category", object], exclude)

    def one_hot_encode(self, df, cols):
        for i in cols:
            dummies = pd.get_dummies(df[i], prefix=i, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(i, axis=1)

        return df

    def split_x_y(self, df):
        X = df.loc[:, df.columns != self.TARGET]
        if len(X.columns) == 1:
            X = X.iloc[:,0]
        y = df.loc[:, self.TARGET]
        return X, y

    def flatten(self, pipe):
        flat = []
        for i in pipe:
            if isinstance(i, list):
                flat.extend(self.flatten(i))
            else:
                flat.append(i)
        return flat

    def pipe_steps(self, pipe):
        return self.flatten(
            [
                x[0]
                if not isinstance(x[1], ColumnTransformer)
                else [
                    list(i[1].named_steps.keys())
                    for ind, i in enumerate(x[1].transformers)
                ]
                for x in pipe.steps
            ]
        )

    def score(
        self, model, scores, stop, start, pipe, note="", all_scores=pd.DataFrame()
    ):
        if len(all_scores) == 0:
            all_scores = pd.DataFrame(
                columns=[
                    "Model",
                    "Mean",
                    "CV Score",
                    "Time",
                    "Cumulative",
                    "Pipe",
                    "Steps",
                    "Note",
                ]
            )

        if len(scores[scores > 0]) == 0:
            note = "Warning: All scores negative"
            mean = 0
            std = 0
        else:
            mean = np.mean(scores[scores > 0])
            std = np.std(scores[scores > 0])

        cumulative = stop - start
        if len(all_scores[all_scores["Model"] == model]) > 0:
            cumulative += (
                all_scores[all_scores["Model"] == model].tail(1)["Cumulative"].values[0]
            )

        return all_scores.append(
            {
                "Model": model,
                "Mean": mean,
                "CV Score": "{:.3f} +/- {:.3f}".format(mean, std),
                "Time": stop - start,
                "Cumulative": cumulative,
                "Pipe": pipe,
                "Steps": ", ".join(self.pipe_steps(pipe)[:-1]),
                "Note": note,
            },
            ignore_index=True,
        )

    def show_scores(self, all_scores, top=False):
        pd.set_option("max_colwidth", -1)

        if top:
            a_s = (
                all_scores.sort_values(["Mean"], ascending=False)
                .groupby("Model")
                .first()
            )
            display(a_s.loc[:, ~a_s.columns.isin(["Mean", "Pipe", "Cumulative"])])
        else:
            display(
                all_scores.loc[
                    :, ~all_scores.columns.isin(["Mean", "Pipe", "Cumulative"])
                ]
            )
        pd.reset_option('max_colwidth', silent=True)

    def plot_models(self, all_scores):
        sns.set_style("whitegrid")
        plt.figure(figsize=(16, 8))
        ax = sns.lineplot(
            x="Cumulative",
            y="Mean",
            hue="Model",
            style="Model",
            markers=True,
            dashes=False,
            data=all_scores,
        )
        label = str(self.METRIC) + " Score"
        ax.set(ylabel=label, xlabel="Time")

    def top_pipeline(self, all_scores, index=0):
        return all_scores.sort_values(by=["Mean"], ascending=False).iloc[index]["Pipe"]

    # Regression

    def scatter_predict(self, y, pred):
        plt.figure(figsize=(11, 9))
        plt.scatter(y, pred, alpha=0.3)
        plt.ylabel("Predicted")
        plt.show()

    def plot_predict(
        self, y, pred, group=None, subset=None, x_label=None, y_label=None
    ):
        if group:
            y = pd.DataFrame(y)
            y = y.groupby(y.index // group).sum().iloc[:, 0]
            pred = pd.DataFrame(pred)
            pred = pred.groupby(pred.index // group).sum().iloc[:, 0]

        if not subset:
            subset = len(y)

        plt.figure(figsize=(19, 9))
        ax = sns.lineplot(data=y[0:subset], color="blue", label="Actual")
        ax = sns.lineplot(data=pred[0:subset], color="red", label="Predicted")

        if x_label:
            plt.xlabel(x_label)

        if y_label:
            plt.ylabel(y_label)

    # Classification

    def plot_roc(self, fpr, tpr, logit_roc_auc):
        plt.figure(figsize=(12, 6))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.show()

    def roc(self, df, model, predictions):
        X, y = self.split_x_y(df)
        logit_roc_auc = roc_auc_score(y, predictions)
        fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:, 1])
        self.plot_roc(fpr, tpr, logit_roc_auc)
        print(classification_report(y, predictions))
