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
    """
    Lớp giảm chiều bằng chọn đặc trưng Chi-Square (SelectKBest + chi2).
    Hỗ trợ hai chế độ như PCA: số đặc trưng cố định (int) hoặc độ chính xác tối thiểu (float).
    """

    def __init__(
        self,
        n_components: Union[int, float],
        random_state: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_components: Số đặc trưng giữ lại (int) hoặc độ chính xác tối thiểu (float trong (0, 1]).
                - int: giữ top-k đặc trưng theo điểm Chi-Square.
                - float: tìm số đặc trưng k nhỏ nhất sao cho classifier đạt ít nhất accuracy = n_components.
            random_state: Seed cho classifier dùng khi n_components là float.
        """
        if isinstance(n_components, int) and n_components <= 0:
            raise ValueError("n_components (int) phải là số nguyên dương.")
        if isinstance(n_components, float) and not (0 < n_components <= 1):
            raise ValueError("n_components (float) phải trong khoảng (0, 1].")
        self._n_components = n_components
        self._random_state = random_state
        self._selector = None
        self._fitted = False
        self._min_accuracy_actual: Optional[float] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ChiSquareReducer":
        """
        Fit selector Chi-Square trên (X, y).
        Dữ liệu X phải không âm (ví dụ MNIST đã chuẩn hóa [0, 1]).

        Args:
            X: (N, D).
            y: Nhãn (N,).
            X_val, y_val: Validation set (chỉ dùng khi n_components là float để tìm k theo accuracy).

        Returns:
            self.
        """
        try:
            from sklearn.feature_selection import SelectKBest, chi2
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError("Cần cài đặt: pip install scikit-learn") from None

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        n_features = X.shape[1]

        if isinstance(self._n_components, int):
            k = min(self._n_components, n_features)
            self._selector = SelectKBest(score_func=chi2, k=k)
            self._selector.fit(X, y)
            self._fitted = True
            return self

        target_acc = float(self._n_components)
        use_val = X_val is not None and y_val is not None
        if use_val and X_val.ndim == 3:
            X_val = X_val.reshape(X_val.shape[0], -1)

        sel_all = SelectKBest(score_func=chi2, k=min(n_features, 784))
        sel_all.fit(X, y)
        scores = sel_all.scores_
        rank = np.argsort(scores)[::-1]

        def accuracy_at_k(k: int) -> float:
            cols = rank[:k]
            Xk = X[:, cols]
            clf = LogisticRegression(max_iter=100, random_state=self._random_state)
            clf.fit(Xk, y)
            if use_val:
                Xv = X_val[:, cols]
                return float(clf.score(Xv, y_val))
            return float(clf.score(Xk, y))

        lo, hi = 1, n_features
        best_k = n_features
        while lo <= hi:
            mid = (lo + hi) // 2
            acc = accuracy_at_k(mid)
            if acc >= target_acc:
                best_k = mid
                hi = mid - 1
            else:
                lo = mid + 1

        self._min_accuracy_actual = accuracy_at_k(best_k)
        self._selector = SelectKBest(score_func=chi2, k=best_k)
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

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit và transform trong một bước."""
        return self.fit(X, y, X_val=X_val, y_val=y_val).transform(X)

    @property
    def n_components_(self) -> int:
        """Số chiều sau khi giảm (số đặc trưng được chọn)."""
        if self._selector is None:
            raise RuntimeError("Chưa gọi fit(X, y).")
        return int(self._selector.get_support().sum())

    @property
    def scores_(self) -> np.ndarray:
        """Điểm Chi-Square theo từng đặc trưng (sau khi fit)."""
        if self._selector is None:
            raise RuntimeError("Chưa gọi fit(X, y).")
        return self._selector.scores_

    def min_accuracy_reached(self) -> Optional[float]:
        """Độ chính xác đạt được khi dùng chế độ float (chỉ có sau fit với n_components float)."""
        return self._min_accuracy_actual
