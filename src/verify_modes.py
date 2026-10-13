"""Formally verify every mode of a trained multi-mode controller with alpha-beta-CROWN,
using the SAME Lyapunov function and the SAME sublevel set {V <= rho} for all modes.

If all modes verify, {V <= rho} is a common invariant set on which V decreases for every
mode, i.e. the closed loop is stable under ARBITRARY switching among the modes.

Usage:
  python src/verify_modes.py --run runs/pendulum_2mode --rho 1.5 [--limit 12 12] [--hole 0.001]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LYA = ROOT / "external" / "Lyapunov_Stable_NN_Controllers"
ABC = ROOT / "external" / "alpha-beta-CROWN" / "complete_verifier"

CONFIG_TEMPLATE = """general:
  conv_mode: matrix
  enable_incomplete_verification: false
  root_path: {root}
  csv_name: specs/spec.csv
model:
  name: >-
    Customized("{factory}", "{factory_fn}", {mode_kw}, dt={dt},
    lyapunov_parameters={{"hidden_widths": [16,16,8], "R_rows": 3, "absolute_output": True, "eps": {veps}, "activation": torch.nn.LeakyReLU, "V_psd_form": "L1"}},
    controller_parameters={{"nlayer": {nlayer}, "hidden_dim": {hidden}, "clip_output":"clamp", "u_lo": torch.tensor([-{umax}]), "u_up": torch.tensor([{umax}])}},
    loss_parameters={{"kappa": {kappa}}},
    loss_func="lyapunov.LyapunovDerivativeSimpleLossWithVBox")
  path: {ckpt}
  input_shape: [-1, 2]
data:
  dataset: Customized("{models_py}", "box_data", lower_limit=[{lo0}, {lo1}], upper_limit=[{up0}, {up1}], ndim=2, scale=1.0, hole_size=1.0)
attack:
  pgd_order: skip
solver:
  batch_size: 100000
  min_batch_size_ratio: 0.
  bound_prop_method: crown
bab:
  override_timeout: {timeout}
  decision_thresh: -1.e-6
  branching:
    method: {branching}
    input_split:
      enable: True
      ibp_enhancement: True
      compare_with_old_bounds: True
      adv_check: -1
      sb_coeff_thresh: 0.001
"""


def env():
    e = os.environ.copy()
    e["PYTHONPATH"] = os.pathsep.join([str(LYA), str(ABC)])
    e["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--limit", type=float, nargs=2, default=None)
    ap.add_argument("--hole", type=float, default=0.001)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--kappa", type=float, default=None, help="override decrease rate in the certificate")
    ap.add_argument("--branching", default="sb", choices=["sb", "naive"])
    ap.add_argument("--modes", type=int, nargs="*", default=None, help="subset of modes to verify")
    args = ap.parse_args()

    run = (ROOT / args.run).resolve()
    targs = json.loads((run / "args.json").read_text())
    limit = args.limit or [targs["limit"][0] * targs["limit_scales"][-1], targs["limit"][1] * targs["limit_scales"][-1]]
    vdir = run / (f"verify_rho{args.rho:g}" + (f"_k{args.kappa:g}" if args.kappa is not None else ""))
    (vdir / "specs").mkdir(parents=True, exist_ok=True)

    # VNNLIB specs (shared by all modes since V, rho and the box are shared)
    subprocess.run([sys.executable, "-m", "neural_lyapunov_training.generate_vnnlib",
                    "--lower_limit", str(-limit[0]), str(-limit[1]),
                    "--upper_limit", str(limit[0]), str(limit[1]),
                    "--hole_size", str(args.hole), "--value_levelset", str(args.rho),
                    str(vdir / "specs" / "spec")], cwd=LYA, env=env(), check=True,
                   stdout=subprocess.DEVNULL)
    # make csv paths relative to root_path
    csv = vdir / "specs" / "spec.csv"
    csv.write_text("".join(f"specs/{Path(l.strip()).name}\n" for l in csv.read_text().splitlines() if l.strip()))

    summary = {"rho": args.rho, "kappa": args.kappa if args.kappa is not None else targs["kappa"], "limit": limit, "modes": []}
    if targs.get("setpoints"):
        mode_list = [("create_pendulum_setpoint", f"theta_star={t}", t) for t in targs["setpoints"]]
    else:
        mode_list = [("create_pendulum_mode", f"mass={ms}", ms) for ms in targs["masses"]]
    for m, (factory_fn, mode_kw, mass) in enumerate(mode_list):
        if args.modes is not None and m not in args.modes:
            continue
        cfg = vdir / f"mode{m}.yaml"
        cfg.write_text(CONFIG_TEMPLATE.format(
            root=vdir.as_posix(), factory=(ROOT / "src" / "verify_factory.py").as_posix(), factory_fn=factory_fn, mode_kw=mode_kw,
            dt=targs["dt"], nlayer=targs["nlayer"], hidden=targs["hidden"], umax=targs["u_max"],
            kappa=args.kappa if args.kappa is not None else targs["kappa"], veps=targs.get("V_eps", 0.01), ckpt=(run / f"verify_mode{m}.pth").as_posix(),
            models_py=(LYA / "neural_lyapunov_training" / "models.py").as_posix(),
            branching=args.branching, lo0=-limit[0], lo1=-limit[1], up0=limit[0], up1=limit[1], timeout=args.timeout))
        logf = vdir / f"mode{m}.log"
        with open(logf, "w") as f:
            subprocess.run([sys.executable, "abcrown.py", "--config", str(cfg)], cwd=ABC, env=env(),
                           stdout=f, stderr=subprocess.STDOUT)
        txt = logf.read_text(errors="ignore")
        safe = re.search(r"total verified \(safe/unsat\): (\d+)", txt)
        unsafe = re.search(r"total falsified \(unsafe/sat\): (\d+)", txt)
        tout = re.search(r"total falsified \(unsafe/sat\): \d+ , timeout: (\d+)", txt)
        mt = re.search(r"mean time for ALL instances \(total \d+\):([\d.]+), max time: ([\d.]+)", txt)
        rec = dict(mode=m, param=mass,
                   safe=int(safe.group(1)) if safe else None,
                   falsified=int(unsafe.group(1)) if unsafe else None,
                   timeout=int(tout.group(1)) if tout else None,
                   mean_time=float(mt.group(1)) if mt else None, max_time=float(mt.group(2)) if mt else None)
        summary["modes"].append(rec)
        print(json.dumps(rec), flush=True)
    n_regions = len([l for l in csv.read_text().splitlines() if l.strip()])
    summary["n_regions"] = n_regions
    summary["all_verified"] = all(r["safe"] == n_regions for r in summary["modes"])
    tag = "" if args.modes is None else "_modes" + "-".join(map(str, args.modes))
    (vdir / f"summary{tag}.json").write_text(json.dumps(summary, indent=2))
    print("ALL MODES VERIFIED (common V, common rho):", summary["all_verified"])


if __name__ == "__main__":
    main()
