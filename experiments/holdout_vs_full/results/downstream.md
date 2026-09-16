# Holdout vs Full CE — downstream

## Summary

| Protocol | α* (θ2) | θ1 | ΔCE (calib) | RWQA exact_match | MMMU mmmu_acc |
|----------|---------|----|-------------|------------------|---------------|
| **A** no cut (Hessian+ΔCE on 128) | **0.30** | 0.70 | 0.019385 | **0.6601** ±0.0171 | **0.4933** |
| **B** hold-out (Hess 96 / ΔCE 32) | **1.00** | 0.00 | 0.017556 | **0.6588** ±0.0172 | **0.4933** |

Judge by downstream metrics, not by comparing A/B ΔCE across protocols.
