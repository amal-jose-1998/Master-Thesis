import numpy as np


# =============================================================================
# Transition matrix builder 
# =============================================================================
def build_structured_transition_params(hdv_dbn):
    """
    Extract structured transition parameters from the pgmpy DBN CPDs.
    We represent:
        pi_s0[s]              = P(s_0 = s)
        pi_a0_given_s0[s,a]  = P(a_0 = a | s_0 = s)
        A_s[s_prev, s]       = P(s_t = s | s_{t-1} = s_prev)
        A_a[s, a_prev, a]    = P(a_t = a | a_{t-1} = a_prev, s_t = s)

    Parameters
    hdv_dbn : HDVDBN
        DBN model with CPDs for Style and Action at time 0 and 1.

    Returns
    pi_s0 : np.ndarray, shape (S,)
    pi_a0_given_s0 : np.ndarray, shape (S, A)
    A_s : np.ndarray, shape (S, S)
    A_a : np.ndarray, shape (S, A, A)
        Interpreted as A_a[s_cur, a_prev, a_cur].
    """
    S = int(hdv_dbn.num_style)
    A = int(hdv_dbn.num_action)

    cpd_style0 = hdv_dbn.model.get_cpds(("Style", 0))
    cpd_action0 = hdv_dbn.model.get_cpds(("Action", 0))
    cpd_style1 = hdv_dbn.model.get_cpds(("Style", 1))
    cpd_action1 = hdv_dbn.model.get_cpds(("Action", 1))

    # -----------------------------
    # pi_s0
    # -----------------------------
    pi_s0 = np.asarray(cpd_style0.values, dtype=float).reshape(S)  # (S,)
    pi_s0_sum = float(pi_s0.sum())
    if not np.isfinite(pi_s0_sum) or pi_s0_sum <= 0.0: # If the CPD is corrupt (NaN/Inf) or sums to 0 (invalid), fall back to uniform.
        pi_s0 = np.full((S,), 1.0 / S, dtype=float)
    else: # normalize to sum to 1.
        pi_s0 = pi_s0 / pi_s0_sum

    # -----------------------------
    # pi_a0_given_s0
    # cpd_action0.values is typically (A, S) = P(a0 | s0) with rows=a, cols=s
    # We transpose to (S, A).
    # -----------------------------
    P_a0_given_s0 = np.asarray(cpd_action0.values, dtype=float).reshape(A, S).T  # (S,A)
    # Normalize each style row so probabilities across actions sum to 1.
    row = P_a0_given_s0.sum(axis=1, keepdims=True)
    row[row <= 0.0] = 1.0
    pi_a0_given_s0 = P_a0_given_s0 / row

    # -----------------------------
    # A_s
    # cpd_style1.values is (S_new, S_old). Convert to (S_old, S_new) and normalize rows.
    # -----------------------------
    P_snew_given_sold = np.asarray(cpd_style1.values, dtype=float).reshape(S, S)  # rows=new, cols=old
    A_s = P_snew_given_sold.T  # (s_old, s_new) => (s_prev, s)
    row = A_s.sum(axis=1, keepdims=True) # Row-normalize so each s_prev row sums to 1.
    row[row <= 0.0] = 1.0 
    A_s = A_s / row

    # -----------------------------
    # A_a
    # CPD: P(Action_1 | Action_0, Style_1)
    #
    # We reshape to vals[a_cur, a_prev, s_cur] and then fill:
    #   A_a[s_cur, a_prev, a_cur] = vals[a_cur, a_prev, s_cur]
    # (then normalize over a_cur per (s_cur, a_prev)).
    # -----------------------------
    vals = np.asarray(cpd_action1.values, dtype=float)
    # pgmpy may provide (A_cur, A_prev*S) or already multi-dim; handle both robustly
    if vals.ndim == 2:
        # (A_cur, A_prev*S) -> (A_cur, A_prev, S)
        vals = vals.reshape(A, A, S)
    elif vals.ndim == 3:
        # expected (A_cur, A_prev, S)
        if vals.shape != (A, A, S):
            vals = vals.reshape(A, A, S)
    else:
        # very unexpected; last-resort reshape
        vals = vals.reshape(A, A, S)

    A_a = np.zeros((S, A, A), dtype=float)  # (s_cur, a_prev, a_cur)
    for s_cur in range(S):
        for a_prev in range(A):
            probs = vals[:, a_prev, s_cur]  # (A_cur,)
            psum = float(probs.sum())
            if not np.isfinite(psum) or psum <= 0.0: # corrupt CPD or sums to 0; fall back to uniform
                A_a[s_cur, a_prev, :] = 1.0 / A
            else:
                A_a[s_cur, a_prev, :] = probs / psum # Normalize the conditional distribution over a_cur

    return pi_s0, pi_a0_given_s0, A_s, A_a