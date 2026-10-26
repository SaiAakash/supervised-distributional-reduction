import torch

from local_ot.gromov._semirelaxed import (
    semirelaxed_gromov_wasserstein,
    entropic_semirelaxed_gromov_wasserstein,
)

from src.affinities import GramAffinity
from src.sdr import SDR, WrongParameter


class DistRCKA(SDR):
    """
    Baseline method: DistR (pure Gromov-Wasserstein coupling) + CKA supervision.

    This is a deliberate ablation of :class:`~src.sdr.SDR`. Whereas SDR uses
    *Fused* Gromov-Wasserstein (FGW) to fold label information directly into the
    transport plan ``T`` through a linear supervision cost ``M``, this baseline
    keeps the coupling *purely structural* -- it is computed with (semi-relaxed)
    Gromov-Wasserstein exactly as in :class:`~src.clust_dr.DistR`, with no label
    term. Supervision enters the model only through a CKA/HSIC term added on top
    of the embedding objective.

    Concretely, the two methods differ in exactly one place, the coupling update:

    ==================  ============================================  ===========================
    Method              Transport plan ``T`` update                   Embedding objective
    ==================  ============================================  ===========================
    SDR                 (semi-relaxed) *Fused* GW  (labels in ``M``)  ``alpha*GW - lambda*CKA``
    DistRCKA (baseline) (semi-relaxed) GW          (unsupervised)      ``alpha*GW - lambda*CKA``
    ==================  ============================================  ===========================

    Because the embedding objective is inherited unchanged from SDR (with
    ``objective="sdr"``), everything else -- the CKA/HSIC supervision term, the
    optional learned projection map ``L`` for out-of-sample ``transform``, loss
    logging and early stopping -- behaves identically to SDR. Only the coupling
    is unsupervised.

    Parameters
    ----------
    alpha : float, optional
        Weight on the structural (GW) term of the embedding objective, by
        default 1.0. Unlike SDR this is *not* an FGW interpolation parameter --
        the coupling is always pure GW -- so any ``alpha > 0`` is valid.
    lambda_hsic : float, optional
        Weight on the CKA/HSIC supervision term, by default 0.5.

    See :class:`~src.sdr.SDR` for the full list of shared parameters.
    """

    def __init__(
        self,
        alpha=1.0,
        affinity_data=GramAffinity(centering=True),
        affinity_embedding=GramAffinity(),
        hsic_kernel_z="rbf",
        hsic_kernel_y="rbf",
        use_rff=False,
        n_features=256,
        output_sam=10,
        output_dim=2,
        loss_fun="square_loss",
        optimizer="Adam",
        lr=1.0,
        init="normal",
        verbose=True,
        tol=1e-6,
        max_iter=1000,
        seed=0,
        device="cpu",
        dtype=torch.double,
        init_T="spectral",
        max_iter_outer=10,
        marginal_loss=False,
        accept_reject_T=True,
        t_gate="J",
        entropic_reg=0.0,
        early_stopping=True,
        lambda_hsic=0.5,
        beta=0.5,
        learn_L=True,
    ):
        if alpha <= 0:
            raise WrongParameter("alpha must be positive (it weights the structural GW term).")

        super(DistRCKA, self).__init__(
            alpha=alpha,
            affinity_data=affinity_data,
            affinity_embedding=affinity_embedding,
            hsic_kernel_z=hsic_kernel_z,
            hsic_kernel_y=hsic_kernel_y,
            use_rff=use_rff,
            n_features=n_features,
            # The baseline is CKA-on-top-of-GW only; the FGW linear-cost path is
            # never used, so we pin the objective accordingly.
            fgw_loss="square_loss",
            output_sam=output_sam,
            output_dim=output_dim,
            loss_fun=loss_fun,
            objective="sdr",
            optimizer=optimizer,
            lr=lr,
            init=init,
            verbose=verbose,
            tol=tol,
            max_iter=max_iter,
            seed=seed,
            device=device,
            dtype=dtype,
            init_T=init_T,
            max_iter_outer=max_iter_outer,
            marginal_loss=marginal_loss,
            accept_reject_T=accept_reject_T,
            t_gate=t_gate,
            entropic_reg=entropic_reg,
            early_stopping=early_stopping,
            lambda_hsic=lambda_hsic,
            beta=beta,
            learn_L=learn_L,
        )

    def _solver_objective(self, T, PZ, M):
        """
        ``F(T)`` for this class's coupling update: pure (semi-relaxed) GW.

        Mirrors :meth:`_solve_T`, which passes ``marginal_loss=self.marginal_loss`` to the srGW
        solver and supplies no linear cost. ``M`` is always None here and is accepted only to
        match the base-class signature.

        The CKA term is added on top by :meth:`SDR._t_gate_objective` when ``t_gate="J"``, so the
        gate covers the complete objective here too even though the solver ignores supervision
        entirely -- which is precisely the F/J gap this baseline is designed to isolate.
        """
        return self._complete_gw_value(PZ, T, marginal_loss=self.marginal_loss)

    def _solve_T(self, return_context=False):
        """
        Compute the (semi-relaxed) Gromov-Wasserstein transport plan.

        This overrides :meth:`SDR._solve_T`, which uses Fused GW. Here the
        coupling is purely structural -- no label-derived linear cost enters --
        matching the unsupervised GW update in :class:`~src.clust_dr.DistR`.

        Only the solver is overridden, so the accept/reject safeguard implemented in
        :meth:`SDR._update_T` still guards this candidate plan.
        """
        PZ = self.affinity_embedding.compute_affinity(self.Z).detach()

        if self.entropic_reg == 0.0:
            T_cand = semirelaxed_gromov_wasserstein(
                self.PX,
                PZ,
                p=self.h0,
                loss_fun=self.loss_fun,
                G0=self.T,
                marginal_loss=self.marginal_loss,
                log=False,
                verbose=False,
            ).detach()
        else:
            T_cand = entropic_semirelaxed_gromov_wasserstein(
                self.PX,
                PZ,
                self.h0,
                loss_fun=self.loss_fun,
                epsilon=self.entropic_reg,
                G0=self.T,
                marginal_loss=self.marginal_loss,
                objective="exact",
                stop_criterion="loss",
                stop_timestep=1,
                log=False,
                verbose=False,
            ).detach()

        return (T_cand, PZ, None) if return_context else T_cand
