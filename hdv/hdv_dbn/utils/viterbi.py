"""
Viterbi decoding for the structured DBN with hidden nodes:
  s_t in {0..S-1}   (style)
  a_t in {0..A-1}   (action)

Transitions (factorized):
  P(s_t | s_{t-1})       = A_s[s_prev, s]
  P(a_t | a_{t-1}, s_t)  = A_a[s, a_prev, a]

Emissions:
  logB[t,s,a] = log p(o_t | s_t=s, a_t=a)

This returns the MAP latent trajectory (Viterbi path) for each sequence.
"""

from dataclasses import dataclass
import torch

@dataclass
class ViterbiResult:
    s_path: torch.Tensor # (T,)
    a_path: torch.Tensor # (T,)

@torch.no_grad()
def viterbi(pi_s0, pi_a0_given_s0, A_s, A_a, logB_s_a):
    eps = 1e-6
    device = logB_s_a.device
    dtype = logB_s_a.dtype
    T, S, A = logB_s_a.shape

    if pi_s0.shape != (S,):
        raise ValueError(f"pi_s0 shape {tuple(pi_s0.shape)} != ({S},)")
    if pi_a0_given_s0.shape != (S, A):
        raise ValueError(f"pi_a0_given_s0 shape {tuple(pi_a0_given_s0.shape)} != ({S},{A})")
    if A_s.shape != (S, S):
        raise ValueError(f"A_s shape {tuple(A_s.shape)} != ({S},{S})")
    if A_a.shape != (S, A, A):
        raise ValueError(f"A_a shape {tuple(A_a.shape)} != ({S},{A},{A})")
    
    # log probs with stability epsilon
    log_pi_s0 = torch.log(pi_s0.to(device=device, dtype=dtype) + eps)                 # (S,)
    log_pi_a0_given_s0 = torch.log(pi_a0_given_s0.to(device=device, dtype=dtype) + eps)        # (S,A)
    logAs = torch.log(A_s.to(device=device, dtype=dtype) + eps)                       # (S,S)
    logAa = torch.log(A_a.to(device=device, dtype=dtype) + eps)                       # (S,A,A)

    # delta[t,s,a] = max log prob of best path ending at (s,a) at time t 
    delta = torch.empty((T, S, A), device=device, dtype=dtype)

    # backpointers for each time t>=1 and each current (s,a); they store how we reached the best state.
    bp_s = torch.empty((max(T - 1, 0), S, A), device=device, dtype=torch.long)  # best previous style index s_prev
    bp_a = torch.empty((max(T - 1, 0), S, A), device=device, dtype=torch.long)  # best previous action index a_prev

    # t=0 init
    delta[0] = log_pi_s0[:, None] + log_pi_a0_given_s0 + logB_s_a[0]  # (S,A)

    if T == 1: # If the sequence has only one timestep:
        # Find best (s,a) at time 0
        flat = delta[0].reshape(-1)
        best_idx = torch.argmax(flat)
        best_s = best_idx // A
        best_a = best_idx % A
        return ViterbiResult(
            s_path=best_s.view(1),
            a_path=best_a.view(1)
        )
    
    for t in range(1, T):
        # The goal for Viterbi recursion is:
        #   δ_t(s, a) = logB[t,s,a] + max_{s_prev, a_prev}(
        #                                                   δ_{t-1}(s_prev, a_prev) + log p(s_t=s | s_{t-1}=s_prev) + log p(a_t=a | a_{t-1}=a_prev, s_t=s)
        #                                                 )
        
        prev = delta[t - 1]  # (S_prev, A_prev), holds all best scores for time t-1.

        # -----------------------------
        # 1) Build scores for action-transition part (plus previous delta)
        # -----------------------------
        # the best score of a path that:
        #   - ended at (s_prev, a_prev) at time t-1
        #   - then transitions to action a at time t, assuming the current style is s
        scores_ap = prev[:, :, None, None] + logAa[None, :, :, :] # (S_prev, S, A_prev, A); δ_{t-1}(s_prev, a_prev) + log p(a_t = a | a_{t-1} = a_prev, s_t = s)
        # -----------------------------
        # 2) Max over previous action a_prev 
        # -----------------------------
        m_val, m_arg = torch.max(scores_ap, dim=2)  # (S_prev, S, A); m_arg stores which a_prev gave the max

        # -----------------------------
        # 3) Add style transition log p(s | s_prev)
        # -----------------------------
        # This is now the best score that uses:
        #   - best previous action a_prev (already maximized)
        #   - and includes the style transition term
        scores_sp = m_val + logAs[:, :, None]  # # (S_prev, S, A)
        # -----------------------------
        # 4) Max over previous style s_prev
        # -----------------------------
        # best_val gives the best score ending at (s,a)
        # best_s_prev tells which previous style was best
        best_val, best_s_prev = torch.max(scores_sp, dim=0) # (S, A)

        # recover best previous action
        best_s_prev_exp = best_s_prev.unsqueeze(0)  # (1,S,A)
        best_a_prev = torch.gather(m_arg, dim=0, index=best_s_prev_exp).squeeze(0)  # (S,A); best_a_prev[s,a] = best previous action index

        # store backpointers for this transition (at index t-1)
        bp_s[t - 1] = best_s_prev
        bp_a[t - 1] = best_a_prev

        # update delta
        delta[t] = logB_s_a[t] + best_val  # (S,A)

    # Termination: pick the best final (s_T, a_T) pair
    flatT = delta[T - 1].reshape(-1)
    best_joint = torch.argmax(flatT)
    sT = (best_joint // A).to(torch.long)
    aT = (best_joint % A).to(torch.long)

    # Backtrack
    s_path = torch.empty((T,), device=device, dtype=torch.long)
    a_path = torch.empty((T,), device=device, dtype=torch.long)

    s_path[T - 1] = sT
    a_path[T - 1] = aT

    s_cur = sT
    a_cur = aT
    for t in range(T - 2, -1, -1):
        # Start from the last state and walk backward using backpointers:
        s_prev = bp_s[t, s_cur, a_cur]
        a_prev = bp_a[t, s_cur, a_cur]
        s_path[t] = s_prev
        a_path[t] = a_prev
        s_cur = s_prev
        a_cur = a_prev

    return ViterbiResult(
        s_path=s_path,
        a_path=a_path
    )


@torch.no_grad()
def durations_from_path(path_1d):
    """
    Compute run-lengths (durations) from a discrete path.
    path_1d: (T,) long
    returns: (num_runs,) long
    """
    if path_1d.ndim != 1:
        raise ValueError(f"path_1d must be 1D, got {tuple(path_1d.shape)}")
    T = int(path_1d.numel())
    if T == 0:
        return torch.zeros((0,), dtype=torch.long, device=path_1d.device)
    if T == 1:
        return torch.ones((1,), dtype=torch.long, device=path_1d.device)

    x = path_1d
    change = (x[1:] != x[:-1]).nonzero(as_tuple=False).reshape(-1) + 1  # indices where a new run starts
    idx = torch.cat([torch.zeros((1,), device=x.device, dtype=torch.long),
                     change.to(torch.long),
                     torch.tensor([T], device=x.device, dtype=torch.long)])
    return (idx[1:] - idx[:-1]).to(torch.long)