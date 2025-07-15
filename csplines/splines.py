from functools import partial
from jax import jit, lax, vmap

import jax.numpy as jnp


def _check_equidistant_knots(knots) -> bool:
    """
    Checks if knots are equidistants.
    """
    diff = jnp.diff(knots)

    return jnp.allclose(diff, diff[0], 1e-3, 1e-3)


def _check_data_range(x, knots, order: int) -> bool:
    """
    Check that values in x are in the range
    [knots[order], knots[dim(knots) - order - 1]].
    """

    return (
        jnp.min(x) >= knots[order] and jnp.max(x) <= knots[knots.shape[0] - order - 1]
    )


def _check_b_spline_inputs(x, knots, order: int) -> None:
    if not order >= 0:
        raise ValueError("Order must non-negative")
    if not _check_equidistant_knots(knots):
        raise ValueError("Sorted knots are not equidistant")
    if not _check_data_range(x, knots, order):
        raise ValueError(
            f"Data values are not in the range \
                [{knots[order]}, {knots[knots.shape[0] - order - 1]}]"
        )


# @jit
@partial(jit, static_argnums=(1, 2))
def create_equidistant_knots(x, order: int = 3, n_params: int = 20):
    """
    Create equidistant knots for B-Spline of the specified order.

    Some additional info:

    - ``dim(knots) = n_params + order + 1``
    - ``n_params = dim(knots) - order - 1``

    Parameters
    ----------
    x
        The data for which the knots are created.
    order
        A positive integer giving the order of the spline function.
        A cubic spline has order of 3.
    n_params
        Number of parameters of the B-spline.
    """
    epsilon = 0.01

    internal_k = n_params - order + 1

    a = jnp.min(x)
    b = jnp.max(x)

    min_k = a - jnp.abs(a * epsilon)
    max_k = b + jnp.abs(b * epsilon)

    internal_knots = jnp.linspace(min_k, max_k, internal_k)

    step = internal_knots[1] - internal_knots[0]

    left_knots = jnp.linspace(min_k - (step * order), min_k - step, order)
    right_knots = jnp.linspace(max_k + step, max_k + (step * order), order)

    return jnp.concatenate((left_knots, internal_knots, right_knots))


@partial(jit, static_argnums=2)
def _build_basis_vector(x, knots, order: int):
    """
    Builds a vector of length (dim(knots) - order - 1).
    Each entry i is iterativaly updated. At time m,
    the entry i is the evaluation of the basis function
    at the observed value for the m-th order and for the i-th knot.
    The creation of the matrix needs a row-wise (order) loop (f1)
    and a column-wise (knot index) loop (f2).
    """
    k = knots.shape[0] - order - 1
    bv = jnp.full(knots.shape[0] - 1, jnp.nan)

    def basis_per_order(m, bv):
        def basis_per_knot(i, bv):
            def base_case(bv):
                return bv.at[i].set(
                    jnp.where(x >= knots[i], 1.0, 0.0)
                    * jnp.where(x < knots[i + 1], 1.0, 0.0)
                )

            def recursive_case(bv):
                b1 = (x - knots[i]) / (knots[i + m] - knots[i]) * bv[i]
                b2 = (
                    (knots[i + m + 1] - x)
                    / (knots[i + m + 1] - knots[i + 1])
                    * bv[i + 1]
                )

                return bv.at[i].set(b1 + b2)

            return lax.cond(m == 0, base_case, recursive_case, bv)

        return lax.fori_loop(0, k + order, basis_per_knot, bv)

    return lax.fori_loop(0, order + 1, basis_per_order, bv)[:k]


def build_design_matrix_b_spline(x, knots, order: int):
    """
    Builds the design matrix for B-Splines of the specified order
    defined by the knots at the values in x. Instead of applying the recursive
    definition of B-splines, a matrix of (order + 1) rows and (dim(knots) - order - 1)
    columns for each value in x is created. This matrix store the evaluation of
    the basis function at the observed value for the m-th order and for the i-th knot.
    """
    knots = jnp.sort(knots)

    _check_b_spline_inputs(x, knots, order)

    return _build_design_matrix_b_spline_aux(x, knots, order)


@partial(jit, static_argnums=2)
def _build_design_matrix_b_spline_aux(x, knots, order: int):
    """
    Fills the design matrix taking the values in the order-th row and of the first
    (dim(knots) - order - 1) columns from the output of the build_basis_matrix function
    called for each data point.
    """
    return vmap(lambda x: _build_basis_vector(x, knots, order))(x)


def _check_p_spline_matrix_inputs(d: int, r: int) -> None:
    if not d > 0:
        raise ValueError("Matix dimension must be positive")
    if not r >= 0:
        raise ValueError("Difference order must be non-negative")


def build_p_spline_matrix(d: int, r: int = 1):
    """
    Builds (d x d) a penalty matrix with differences of order r.
    """
    _check_p_spline_matrix_inputs(d, r)

    return _build_p_spline_matrix_aux(d, r)


@partial(jit, static_argnums=(0, 1))
def _build_p_spline_matrix_aux(d: int, r: int = 1):
    D = jnp.diff(jnp.identity(d), r, axis=0)

    return D.T @ D


def _constraint_z(c):
    """
    Returns the matrix "Z" for reparameterization based on constraint matrix c.
    The constraint is ``c @ b = 0``, where ``b`` is the coefficient vector.
    """
    m = c.shape[0]
    q, _ = jnp.linalg.qr(c.T, mode='complete')
    return q[:, m:]


def constraint_sumzero(X):
    """Matrix "Z" for reparameterization for sum-to-zero-constraint."""
    j = jnp.ones(shape=(X.shape[0], 1))
    C = j.T @ X
    # C = jnp.expand_dims(jnp.mean(X, axis=0), axis=0)
    return _constraint_z(C)


def center_spline(X, K):
    Z = constraint_sumzero(X)
    X = X @ Z
    K = Z.T @ K @ Z

    return X, K
