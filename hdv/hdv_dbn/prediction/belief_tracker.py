# belief_tracker.py
from dataclasses import dataclass
import torch

EPS = 1e-6

@dataclass
class BeliefState:
    # log p(s,a | o_{0:t}) (normalized)
    log_b: torch.Tensor  # (S,A)

class BeliefTracker:
    """
    Online filtering over z_t=(s_t,a_t) + open-loop rollout using structured transitions.
    Works directly with your trained HDVTrainer-like object (pi_s0, pi_a0_given_s0, A_s, A_a, emissions).
    """
    def __init__(self, *, pi_s0, pi_a0_given_s0, A_s, A_a, emissions):
        self.pi_s0 = pi_s0
        self.pi_a0_given_s0 = pi_a0_given_s0
        self.A_s = A_s
        self.A_a = A_a
        self.emissions = emissions

        self.device = getattr(emissions, "_device", pi_s0.device)
        self.dtype = getattr(emissions, "_dtype", pi_s0.dtype)

        self.pi_s0 = self.pi_s0.to(self.device, self.dtype)
        self.pi_a0_given_s0 = self.pi_a0_given_s0.to(self.device, self.dtype)
        self.A_s = self.A_s.to(self.device, self.dtype)
        self.A_a = self.A_a.to(self.device, self.dtype)

        self.S = int(self.pi_s0.numel())
        self.A = int(self.pi_a0_given_s0.shape[1])

        self._logAs = torch.log(self.A_s + EPS)  # (S,S)
        self._logAa = torch.log(self.A_a + EPS)  # (S,A,A)
        # convenience: (A_prev, S_next, A_next)
        self._logAa_ap_s_an = self._logAa.permute(1, 0, 2).contiguous()  # (A,S,A)

        self.reset()

    def reset(self):
        # log p(s0,a0) = log p(s0) + log p(a0|s0)
        log_b0 = torch.log(self.pi_s0 + EPS)[:, None] + torch.log(self.pi_a0_given_s0 + EPS)  # (S,A)
        log_b0 = log_b0 - torch.logsumexp(log_b0.reshape(-1), dim=0)
        self.state = BeliefState(log_b=log_b0)

    @torch.no_grad()
    def update(self, o_t):
        """
        Filtering update: log_b <- log p(s_t,a_t | o_{0:t})
        """
        # emissions.loglikelihood expects (T,F); here T=1
        o = torch.as_tensor(o_t, device=self.device, dtype=self.dtype).unsqueeze(0)
        logB = self.emissions.loglikelihood(o)[0]  # (S,A)

        log_joint = self.state.log_b + logB
        logZ = torch.logsumexp(log_joint.reshape(-1), dim=0)
        log_b_post = log_joint - logZ
        self.state = BeliefState(log_b=log_b_post)
        return float(logZ.detach().cpu().item())  # log p(o_t | o_{0:t-1})

    @torch.no_grad()
    def predict_next_belief(self):
        """
        One-step belief prediction: log p(s_{t+1},a_{t+1} | o_{0:t})
        """
        log_b = self.state.log_b  # (S,A)
        tmp = (
            log_b[:, :, None, None] +
            self._logAs[:, None, :, None] +
            self._logAa_ap_s_an[None, :, :, :]
        )  # (S,A,S',A')
        log_b_next = torch.logsumexp(tmp, dim=(0, 1))  # (S',A')
        log_b_next = log_b_next - torch.logsumexp(log_b_next.reshape(-1), dim=0)
        return BeliefState(log_b=log_b_next)

    @torch.no_grad()
    def rollout(self, H: int):
        """
        Open-loop rollout of belief for horizons 1..H.
        Returns list[BeliefState] of length H.
        """
        out = []
        st = self.state
        for _ in range(int(H)):
            st = self._predict_from(st)
            out.append(st)
        return out

    def _predict_from(self, st: BeliefState):
        log_b = st.log_b
        tmp = (
            log_b[:, :, None, None] +
            self._logAs[:, None, :, None] +
            self._logAa_ap_s_an[None, :, :, :]
        )
        log_b_next = torch.logsumexp(tmp, dim=(0, 1))
        log_b_next = log_b_next - torch.logsumexp(log_b_next.reshape(-1), dim=0)
        return BeliefState(log_b=log_b_next)
