import math
import torch


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
        qmuon_params: The QK parameters to be optimized by qMuon (coupled polar factor updates).
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        qmuon_params=None,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        qmuon_params = list(qmuon_params) if qmuon_params is not None else []
        params.extend(adamw_params)
        params.extend(qmuon_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, qMuon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
            self.state[p]["use_qmuon"] = False
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False
            self.state[p]["use_qmuon"] = False
        for p in qmuon_params:
            # Use qMuon for QK parameters
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = False
            self.state[p]["use_qmuon"] = True
        
        # Store QK pairs for coupled updates
        self.qk_pairs = self._build_qk_pairs(qmuon_params) if qmuon_params else []

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def _build_qk_pairs(self, qmuon_params):
        """Build QK parameter pairs for coupled updates."""
        # Simple heuristic: pair by index (assumes QK parameters are ordered)
        pairs = []
        for i in range(0, len(qmuon_params), 2):
            if i + 1 < len(qmuon_params):
                pairs.append((qmuon_params[i], qmuon_params[i + 1]))
        
        return pairs

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #         qMuon QK         #
            ############################

            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # Apply coupled QK updates using stored pairs
            for wq, wk in self.qk_pairs:
                gq, gk = wq.grad, wk.grad
                
                if gq is None or gk is None:
                    continue
                
                if gq.ndim > 2:
                    gq = gq.view(gq.size(0), -1)
                if gk.ndim > 2:
                    gk = gk.view(gk.size(0), -1)
                
                # Approximate GM = X @ GS @ X.T using gradient coupling
                # This is a heuristic approximation of the bilinear structure
                with torch.no_grad():
                    gm_q = wk.T @ gq  # Approximate X @ GS @ X.T for Q
                    gm_k = wq.T @ gk  # Approximate X @ GS @ X.T for K
                    
                    # Use average of both directions for symmetry
                    gm = (gm_q + gm_k.T) / 2
                    gm = gm.to(dtype=wq.dtype)
                    
                    # Momentum on GM
                    state_q = self.state[wq]
                    if "momentum_buffer" not in state_q:
                        state_q["momentum_buffer"] = torch.zeros_like(gm)
                    
                    buf = state_q["momentum_buffer"]
                    buf.mul_(momentum).add_(gm, alpha=1-momentum)
                    
                    # Apply polar factor to GM
                    p = zeropower_via_newtonschulz5(buf, steps=group["ns_steps"])
                    p = p.to(dtype=wq.dtype)
                    
                    # Coupled updates: WQ = WQ - eta * WK @ P, WK = WK - eta * WQ @ P.T
                    adjusted_lr = self.adjust_lr_for_muon(lr, wq.shape)
                    
                    # Apply weight decay
                    wq.data.mul_(1 - lr * wd)
                    wk.data.mul_(1 - lr * wd)
                    
                    # Apply coupled updates
                    wq.data.add_(wk @ p, alpha=-adjusted_lr)
                    wk.data.add_(wq @ p.T, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"] and not self.state[p]["use_qmuon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1, momentum=0.95, beta1=0.9, beta2=0.95):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2)
        )
    elif optimizer_name == "sgdm":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd, momentum=momentum
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            momentum=momentum,
            adamw_params=adamw_params,
            adamw_betas=(beta1, beta2),
        )
    elif optimizer_name == "qmuon":
        # Separate QK from VO/FFN parameters
        qk_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and ("q_proj" in name or "k_proj" in name or "query" in name or "key" in name)
        ]
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            and not ("q_proj" in name or "k_proj" in name or "query" in name or "key" in name)
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            momentum=momentum,
            adamw_params=adamw_params,
            adamw_betas=(beta1, beta2),
            qmuon_params=qk_params,
        )
    else:
        assert 0, "optimizer not supported"
