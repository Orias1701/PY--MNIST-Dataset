"""Module giảm chiều dữ liệu (PCA, Chi-Square) cho MNIST."""

import numpy as np
from typing import Union, Optional, Tuple


class DimensionalityReducer:
    """Lớp giảm chiều dữ liệu (PCA)."""

    def __init__(
        self,
        n_components: Union[int, float],
        random_state: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_components: Số thành phần giữ lại (int) hoặc tỉ lệ phương sai (float 0–1).
            random_state: Seed cho reproducibility.
        """
        self._n_components = n_components
        self._random_state = random_state
        self._fitted = False
        self._pca = None
        self._n_components_actual: Optional[int] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DimensionalityReducer":
        """
        Fit PCA trên dữ liệu X.

        Args:
            X: (N, D) đã flatten nếu cần.

        Returns:
            self.
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("Cần cài đặt: pip install scikit-learn") from None

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        self._pca = PCA(
            n_components=self._n_components,
            random_state=self._random_state,
        )
        self._pca.fit(X)
        self._n_components_actual = self._pca.n_components_
        self._explained_variance_ratio = self._pca.explained_variance_ratio_
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X sang không gian giảm chiều."""
        if not self._fitted or self._pca is None:
            raise RuntimeError("Chưa gọi fit().")
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self._pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit và transform trong một bước."""
        return self.fit(X).transform(X)

    @property
    def n_components_(self) -> int:
        """Số chiều sau khi giảm."""
        if self._n_components_actual is None:
            raise RuntimeError("Chưa gọi fit().")
        return self._n_components_actual

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Tỉ lệ phương sai được giữ lại theo từng thành phần."""
        if self._explained_variance_ratio is None:
            raise RuntimeError("Chưa gọi fit().")
        return self._explained_variance_ratio

    def total_explained_variance_ratio(self) -> float:
        """Tổng tỉ lệ phương sai được giữ lại."""
        return float(np.sum(self.explained_variance_ratio_))


class ChiSquareReducer:
    """Lớp giảm chiều bằng chọn đặc trưng Chi-Square (SelectKBest + chi2)."""

    def __init__(self, n_components: int) -> None:
        """
        Args:
            n_components: Số đặc trưng giữ lại (top-k theo điểm Chi-Square).
        """
        if n_components <= 0:
            raise ValueError("n_components phải là số nguyên dương.")
        self._n_components = n_components
        self._selector = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChiSquareReducer":
        """
        Fit selector Chi-Square trên (X, y).
        Dữ liệu X phải không âm (ví dụ MNIST đã chuẩn hóa [0, 1]).

        Args:
            X: (N, D).
            y: Nhãn (N,).

        Returns:
            self.
        """
        try:
            from sklearn.feature_selection import SelectKBest, chi2
        except ImportError:
            raise ImportError("Cần cài đặt: pip install scikit-learn") from None

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        k = min(self._n_components, X.shape[1])
        self._selector = SelectKBest(score_func=chi2, k=k)
        self._selector.fit(X, y)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Chọn k đặc trưng từ X."""
        if not self._fitted or self._selector is None:
            raise RuntimeError("Chưa gọi fit(X, y).")
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self._selector.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit và transform trong một bước."""
        return self.fit(X, y).transform(X)

    @property
    def n_components_(self) -> int:
        """Số chiều sau khi giảm (số đặc trưng được chọn)."""
        if self._selector is None:
            raise RuntimeError("Chưa gọi fit(X, y).")
        return self._selector.get_support().sum()

    @property
    def scores_(self) -> np.ndarray:
        """Điểm Chi-Square theo từng đặc trưng (sau khi fit)."""
        if self._selector is None:
            raise RuntimeError("Chưa gọi fit(X, y).")
        return self._selector.scores_
