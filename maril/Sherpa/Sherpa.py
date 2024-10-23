"""
An implementation of the Pearson based KNN method from the following work:

S. Tiku, S. Pasricha, B. Notaros and Q. Han, 
"SHERPA: A Lightweight Smartphone Heterogeneity Resilient Portable Indoor Localization Framework," 
2019 IEEE International Conference on Embedded Software and Systems (ICESS), 2019

Sub-project refers to the implementation of AdLoc

Notes to contributors:
- Keep all dependencies within sub-project
- Usable functions should be importable from outside sub-project
- Enable high flexibility when possible from outside sub-project
- In older versions of sklearn.KNN, the custom metrics arg was broken
  Make sure that issue does not persist in the current implmentation
"""


# TODO:
# Function(s) to build model as per paper
# Ability to provide custom model structure as input

"""
An implementation of the Pearson based KNN method from the following work:

S. Tiku, S. Pasricha, B. Notaros and Q. Han, 
"SHERPA: A Lightweight Smartphone Heterogeneity Resilient Portable Indoor Localization Framework," 
2019 IEEE International Conference on Embedded Software and Systems (ICESS), 2019

Sub-project refers to the implementation of AdLoc

Notes to contributors:
- Keep all dependencies within sub-project
- Usable functions should be importable from outside sub-project
- Enable high flexibility when possible from outside sub-project
- In older versions of sklearn.KNN, the custom metrics arg was broken
  Make sure that issue does not persist in the current implmentation
"""


# TODO:
# Function(s) to build model as per paper
# Ability to provide custom model structure as input

from typing import Iterable
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr


class Sherpa(KNeighborsClassifier):
    """Wrapper on KNN with Pearson Correlation as distance metric
    See original doc below:
    Classifier implementing the k-nearest neighbors vote.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.  Possible values:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.
    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels known to the classifier
    effective_metric_ : str or callble
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.
    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    n_samples_fit_ : int
        Number of samples in the fitted data.
    outputs_2d_ : bool
        False when `y`'s shape is (n_samples, ) or (n_samples, 1) during fit
        otherwise True.
    """

    # TODO: needs testing

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:

        # overwrite metric arg
        kwargs["metric"] = self.perason_distance
        super().__init__(*args, **kwargs)

    def perason_distance(
        self,
        x: Iterable,
        y: Iterable,
    ) -> float:
        """Use PCC as distance

        Parameters
        ----------
        x : Iterable
            Random variable x
        y : Iterable
            Random variable y

        Returns
        -------
        float
            PCC with inverted sign
        """
        r, _ = pearsonr(x, y)
        return 1 - r
