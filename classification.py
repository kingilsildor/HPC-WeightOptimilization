import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import time

DATA_LOCATION = "data/synthetic_delirium_dataset.csv"

open("classification.log", "w").close()
logging.basicConfig(
    filename="classification.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def data_prep() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load and prepare the dataset.

    Return
    ------
    X (pd.DataFrame): feature matrix with categorical values converted to numeric.
    y (np.ndarray): target array containing the `delirium` column values.
    """

    df = pd.read_csv(DATA_LOCATION)
    df = df.replace({"yes": 1, "no": 0})

    col = df.columns[-2]
    df[col] = (
        df[col].replace({"CAM": 1, "MOTYB": 2, "Both": 3, "Neither": 0}).astype(int)
    )

    y = df["delirium"].values
    X = df.drop("delirium", axis=1)
    return X, y


def generate_weights(features: list, seed: int = None) -> pd.DataFrame:
    """
    Create a DataFrame of random weights for a list of features.

    Params
    ------
    features (iterable): sequence of feature names (used for the DataFrame index).
    seed (int or None): random seed for reproducibility. Default is None.

    Return
    ------
    df (pd.DataFrame): DataFrame with columns `feature` and `weight`.
    """

    rng = np.random.default_rng(seed)

    weights = rng.uniform(-1, 1, size=len(features))
    df = pd.DataFrame({"feature": features, "weight": weights})

    return df


def create_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    enable_scaler: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split features and labels into train/test sets and optionally scale them.

    Params
    ------
    X (array-like or pd.DataFrame): feature matrix.
    y (array-like): target values.
    test_size (float): fraction of data to hold out for testing. Default 0.2.
    enable_scaler (bool): whether to apply `StandardScaler` to features. Default True.

    Return
    ------
    X_train, X_test, y_train, y_test : tuple
        Arrays for training and testing features and targets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if enable_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean squared error between true and predicted values.

    Params
    ------
    y_true (array-like): true target values.
    y_pred (array-like): predicted target values.

    Return
    ------
    float: mean squared error.
    """

    return np.mean((y_true - y_pred) ** 2)


def annealing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iters: int = 50000,
    temp_start: float = 1.0,
    temp_end: float = 0.01,
) -> tuple[float, np.ndarray]:
    """
    Simple simulated annealing optimizer to search for feature weights.

    Params
    ------
    X_train (np.ndarray): training feature matrix.
    y_train (np.ndarray): training targets.
    n_iters (int): number of annealing iterations. Default 50000.
    temp_start (float): starting temperature. Default 1.0.
    temp_end (float): final temperature. Default 0.01.

    Return
    ------
    best_mse (float): best mean squared error found on the training set.
    best_weights (np.ndarray): weight vector corresponding to `best_mse`.

    Notes
    -----
    This implementation perturbs the current weights with Gaussian noise
    and accepts worse solutions probabilistically according to the Boltzmann
    criterion controlled by a geometrically decaying temperature.
    """
    logging.info("Starting simulated annealing optimization.")
    logging.info("Number of iterations: %d", n_iters)
    logging.info("Temperature start: %.2f, end: %.2f", temp_start, temp_end)
    logging.info("=" * 50)

    decay = (temp_end / temp_start) ** (1 / n_iters)

    best_weights = weights.copy()
    best_mse = mse(y_train, X_train @ best_weights + intercept)

    current_weights = weights.copy()
    temp = temp_start

    for _ in range(n_iters + 1):
        perturbation = np.random.normal(0, 0.05, size=current_weights.shape)
        new_weights = current_weights + perturbation

        new_mse = mse(y_train, X_train @ new_weights + intercept)

        if new_mse < best_mse or np.random.rand() < np.exp(
            -(new_mse - best_mse) / temp
        ):
            current_weights = new_weights
            if new_mse < best_mse:
                best_weights = new_weights
                best_mse = new_mse

        temp *= decay

        if _ % 1000 == 0:
            logging.info(f"Iteration {_}, Best MSE: {best_mse:.5f}, Temp: {temp:.5f}")
    return best_mse, best_weights


if __name__ == "__main__":
    start_time = time.time()
    logging.info("=" * 50)
    logging.info("Script started at: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("=" * 50)

    X, y = data_prep()
    X_train, X_test, y_train, y_test = create_split(X, y)

    random_weights = generate_weights(X.columns)
    intercept = 0.0
    weights = random_weights.set_index("feature").loc[X.columns].values.flatten()

    best_mse, best_weights = annealing(X_train, y_train)
    train_pred = X_train @ best_weights + intercept
    test_pred = X_test @ best_weights + intercept

    logging.info("=" * 50)
    logging.info("Optimized Train MSE: %f", mse(y_train, train_pred))
    logging.info("Optimized Test MSE: %f", mse(y_test, test_pred))

    optimized_df = pd.DataFrame({"feature": X.columns, "weight": best_weights})
    optimized_df.to_csv("optimized_weights.csv", index=False)

    logging.info("=" * 50)
    logging.info("Script finished at: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    runtime = time.time() - start_time
    logging.info("Total runtime: %.2f seconds", runtime)
    logging.info("=" * 50)
    optimized_df.to_csv("optimized_weights.csv", index=False)
