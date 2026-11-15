"""Every number in manuscript/ comes from here (saved logits, no GPU).

    python analysis/paper_tables.py results

Writes manuscript/tables/*.tex, manuscript/figures/recency_gain.{pdf,png} and
manuscript/numbers.tex (macros for numbers quoted in the text).
Exploratory set: ER / DER++ runs analysed before the confirmation rule (docs/rq1_findings.md).
Confirmatory set: the 14 runs of experiments/plans/s3_confirm.yaml (rule: analysis/confirm_test.py).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "analysis")]
import rq1_calibration as R  # noqa: E402
from recency import recency_share  # noqa: E402
from run import expand_plan  # noqa: E402
from summarize import variant  # noqa: E402
from facecl.config import load_config  # noqa: E402
from facecl.evaluation import metrics as M  # noqa: E402
from facecl.training.trainer import run_id_for  # noqa: E402

OUT = ROOT / "manuscript"
DS = {"seq-cifar10": "S-CIFAR-10", "seq-cifar100": "S-CIFAR-100"}
METH = {"er": "ER", "derpp": "DER++", "mir": "MIR", "icarl": "iCaRL"}
VAR = {"amp": "", "fp32": " (fp32)"}


def collect(results):
    confirm = {run_id_for(load_config(d, m, ov)) for d, m, ov in expand_plan(ROOT / "experiments/plans/s3_confirm.yaml")}
    rows = []
    for rd in sorted(Path(results).glob("seq-*/")):
        if not (rd / "summary.json").exists() or not list((rd / "eval").glob("task_*.npz")):
            continue
        s = json.loads((rd / "summary.json").read_text())
        if s.get("status") != "complete" or s["method"] not in METH:
            continue
        cfg = yaml.safe_load((rd / "config.yaml").read_text())
        _, res = R.evaluate_run(rd)
        share, chance = recency_share(rd)
        mech = mechanism(rd, s)
        rows.append(dict(split="confirm" if rd.name in confirm else "explore", dataset=s["dataset"],
                         method=s["method"], buffer=s["buffer_size"], variant=variant(cfg),
                         seed=s["seed"], share=share, chance=chance, res=res, mech=mech,
                         acc=float(json.loads((rd / "summary.json").read_text()).get("final_avg_acc", np.nan))
                         if "final_avg_acc" in s else np.nan))
    return rows


def mechanism(rd, s):
    """Post-hoc (after S3): fitted temperatures, buffer accuracy, and a trivial predicted-task rule."""
    d = dict(np.load(sorted((rd / "eval").glob("task_*.npz"))[-1]))
    T = s["n_tasks"]
    k = len(s["class_order"]) // T
    keep = ~d["buf_probe"].astype(bool)
    cl = np.concatenate([d["val_logits"], d["buf_logits"][keep]]).astype(float)
    cy = np.concatenate([d["val_y"], d["buf_y"][keep]]).astype(int)
    g = M.fit_temperature(cl, cy)
    tt = R.group_temperatures(cl, None, cy // k, g, cy)
    l = np.concatenate([d[f"test{j}_logits"] for j in range(T)]).astype(float)
    y = np.concatenate([d[f"test{j}_y"] for j in range(T)]).astype(int)
    p = l.argmax(1)
    rule = M.softmax_np(l).max(1) + (p // k < T - 1)  # predicted-old above predicted-current, ties by MSP
    return dict(t_global=g, t_old=float(np.mean([v for j, v in tt.items() if j < T - 1])), t_cur=tt[T - 1],
                buf_acc=float((d["buf_logits"][keep].argmax(1) == d["buf_y"][keep]).mean() * 100),
                rule_auroc=M.auroc_error(rule, p == y) * 100)


def mech_table(rows, path):
    g = defaultdict(list)
    for r in rows:
        g[r["method"]].append(r)
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Why task-TS behaves as it does (post-hoc analysis over all " + str(len(rows)) + r" runs). "
         r"Temperatures fitted on the legal calibration set; $T_\mathrm{old}$ is the mean over old tasks. "
         r"\emph{Rule}: rank predictions in an old task above predictions in the current task, ties broken "
         r"by MSP. Last column: Pearson correlation, over runs, of the AUROC$_\mathrm{all}$ gains of task-TS "
         r"and of the rule.}\label{tab:mech}",
         r"\begin{tabular}{lc c ccc ccc c}\toprule",
         r" & & buffer & \multicolumn{3}{c}{temperature} & \multicolumn{3}{c}{AUROC$_\mathrm{all}$} & \\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
         r"method & $n$ & acc. (\%) & global & $T_\mathrm{old}$ & $T_\mathrm{cur}$ & MSP & task-TS & rule & corr.\\\midrule"]
    for m in ("er", "mir", "derpp", "icarl"):
        rs = g[m]
        f = lambda key: [x["mech"][key] for x in rs]
        gt = np.array([x["res"]["task_ts"]["auroc_all"] - x["res"]["none"]["auroc_all"] for x in rs])
        gr = np.array([x["mech"]["rule_auroc"] - x["res"]["none"]["auroc_all"] for x in rs])
        L.append(f"{METH[m]} & {len(rs)} & {np.mean(f('buf_acc')):.1f} & {np.mean(f('t_global')):.2f} & "
                 f"{np.mean(f('t_old')):.2f} & {np.mean(f('t_cur')):.2f} & "
                 f"{np.mean([x['res']['none']['auroc_all'] for x in rs]):.1f} & "
                 f"{np.mean([x['res']['task_ts']['auroc_all'] for x in rs]):.1f} & {np.mean(f('rule_auroc')):.1f} & "
                 f"{np.corrcoef(gt, gr)[0, 1]:+.2f}" + r"\\")
    L += [r"\bottomrule\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def ms(v, d=1):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f}" if len(v) < 2 else f"{v.mean():.{d}f}{{\\scriptsize$\\pm${v.std(ddof=1):.{d}f}}}"


def label(r):
    v = VAR.get(r["variant"], " (lr decay)" if r["variant"].startswith("amp+lr") else " (" + r["variant"] + ")")
    return f"{DS[r['dataset']]} & {METH[r['method']]}{v} & {r['buffer']}"


def main_table(rows, split, path, caption, lab):
    g = defaultdict(list)
    for r in rows:
        if r["split"] == split:
            g[(r["dataset"], r["method"], r["buffer"], r["variant"])].append(r)
    L = [r"\begin{table*}[t]\centering\small", rf"\caption{{{caption}}}\label{{{lab}}}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{lllc c ccc ccc}\toprule",
         r" & & & & recency & \multicolumn{3}{c}{AUROC$_\mathrm{all}$ ($\uparrow$)} & \multicolumn{3}{c}{ECE$_\mathrm{all}$ ($\downarrow$)}\\",
         r"\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"benchmark & method & $|M|$ & $n$ & share (\%) & MSP & TS & task-TS & MSP & TS & task-TS\\\midrule"]
    for key in sorted(g, key=lambda k: (k[0], k[1] != "er", k[1], k[2], k[3])):
        rs = g[key]
        f = lambda c, m: [x["res"][c][m] for x in rs]
        L.append(f"{label(rs[0])} & {len(rs)} & {ms([x['share'] for x in rs])} & "
                 + " & ".join(ms(f(c, "auroc_all")) for c in ("none", "ts", "task_ts")) + " & "
                 + " & ".join(ms(f(c, "ece_all")) for c in ("none", "ts", "task_ts")) + r"\\")
    L += [r"\bottomrule\end{tabular}}", r"\end{table*}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def oracle_table(rows, path):
    g = defaultdict(list)
    for r in rows:
        g[r["split"]].append(r)
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{Old classes only, task-independent scores (where old-class AUROC is valid). "
         r"Oracle TS is fitted on the evaluated old-class test data itself and is an upper bound for any "
         r"single temperature. Mean over runs; relative ECE change is averaged per run.}\label{tab:oracle}",
         r"\begin{tabular}{lc ccc ccc}\toprule",
         r" & & \multicolumn{3}{c}{ECE$_\mathrm{old}$} & \multicolumn{3}{c}{AUROC$_\mathrm{old}$}\\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"set & $n$ & MSP & TS & oracle TS & MSP & TS & oracle TS\\\midrule"]
    for split, name in (("explore", "exploratory"), ("confirm", "confirmatory")):
        rs = g[split]
        f = lambda c, m: [x["res"][c][m] for x in rs]
        L.append(f"{name} & {len(rs)} & " + " & ".join(ms(f(c, "ece_old")) for c in ("none", "ts", "oracle_ts"))
                 + " & " + " & ".join(ms(f(c, "auroc_old")) for c in ("none", "ts", "oracle_ts")) + r"\\")
    L += [r"\bottomrule\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mk = {"er": "o", "derpp": "s", "mir": "^", "icarl": "D"}
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for split, col in (("explore", "#9aa5b1"), ("confirm", "#1f5fa8")):
        for m in METH:
            rs = [r for r in rows if r["split"] == split and r["method"] == m]
            if rs:
                ax.scatter([r["share"] for r in rs],
                           [r["res"]["task_ts"]["auroc_all"] - r["res"]["none"]["auroc_all"] for r in rs],
                           marker=mk[m], s=28, c=col, edgecolors="k", linewidths=0.3,
                           label=f"{METH[m]} ({'confirmatory' if split == 'confirm' else 'exploratory'})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("old-class errors predicted as a current-task class (%)")
    ax.set_ylabel(r"task-TS $-$ MSP, AUROC$_\mathrm{all}$ (points)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=200)


def numbers(rows, path):
    ex = [r for r in rows if r["split"] == "explore"]
    co = [r for r in rows if r["split"] == "confirm"]
    gain = lambda rs: np.array([r["res"]["task_ts"]["auroc_all"] - r["res"]["none"]["auroc_all"] for r in rs])
    sh = lambda rs: np.array([r["share"] for r in rs])
    rho_ex, p_ex = stats.spearmanr(sh(ex), gain(ex))
    rho_co, p_co = stats.spearmanr(sh(co), gain(co), alternative="greater")
    d_ts = np.array([r["res"]["ts"]["auroc_all"] - r["res"]["none"]["auroc_all"] for r in co])
    tost = max(stats.ttest_1samp(d_ts, -1, alternative="greater").pvalue, stats.ttest_1samp(d_ts, 1, alternative="less").pvalue)
    worse_none = lambda rs: sum(r["res"]["task_ts"]["ece_all"] > r["res"]["none"]["ece_all"] for r in rs)
    worse_ts = lambda rs: sum(r["res"]["task_ts"]["ece_all"] > r["res"]["ts"]["ece_all"] for r in rs)
    rel = np.mean([(r["res"]["none"]["ece_old"] - r["res"]["oracle_ts"]["ece_old"]) / r["res"]["none"]["ece_old"] for r in co]) * 100
    d4 = np.mean([r["res"]["oracle_ts"]["auroc_old"] - r["res"]["none"]["auroc_old"] for r in co])
    by = lambda rs, m: sh([r for r in rs if r["method"] == m])
    # task-dependent score evaluated with the (confounded) old-class-only AUROC, exploratory set
    conf_old = np.mean([r["res"]["task_ts"]["auroc_old"] - r["res"]["none"]["auroc_old"] for r in ex])
    conf_all = np.mean(gain(ex))
    N = {
        "nExplore": len(ex), "nConfirm": len(co), "nTotal": len(rows),
        "rhoExplore": f"{rho_ex:+.2f}", "rhoConfirm": f"{rho_co:+.2f}",
        "pConfirm": f"{p_co:.1e}".replace("e-0", r"\times10^{-") + "}" if p_co < 1e-3 else f"{p_co:.3f}",
        "tsGainConfirm": f"{d_ts.mean():+.2f}", "tostP": f"{tost:.2f}",
        "worseNoneExplore": worse_none(ex), "worseNoneConfirm": worse_none(co),
        "worseTsExplore": worse_ts(ex), "worseTsConfirm": worse_ts(co),
        "oracleEceRel": f"{rel:.0f}", "oracleAurocGain": f"{d4:+.2f}",
        "shareER": f"{min(by(rows, 'er')):.0f}--{max(by(rows, 'er')):.0f}",
        "shareDERpp": f"{min(by(rows, 'derpp')):.0f}--{max(by(rows, 'derpp')):.0f}",
        "shareMIR": f"{min(by(rows, 'mir')):.0f}--{max(by(rows, 'mir')):.0f}",
        "shareiCaRL": f"{min(by(rows, 'icarl')):.0f}--{max(by(rows, 'icarl')):.0f}",
        "confoundOld": f"{conf_old:+.1f}", "confoundAll": f"{conf_all:+.1f}",
    }
    path.write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in N.items()), encoding="utf-8")
    return N


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results")
    args = ap.parse_args(argv)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    rows = collect(args.results)
    main_table(rows, "confirm", OUT / "tables/confirm.tex",
               "Confirmatory runs (never analysed before the rule was committed), state after the last task, all seen "
               "classes. Recency share: old-class errors predicted as a current-task class (uniform errors: 10\\%). "
               "Mean{\\scriptsize$\\pm$sd} over seeds.", "tab:confirm")
    main_table(rows, "explore", OUT / "tables/explore.tex",
               "Exploratory runs (used to form the hypotheses), same quantities as Table~\\ref{tab:confirm}. "
               "Uniform-error recency baseline: 22\\% on S-CIFAR-10, 10\\% on S-CIFAR-100.", "tab:explore")
    oracle_table(rows, OUT / "tables/oracle.tex")
    mech_table(rows, OUT / "tables/mech.tex")
    figure(rows, OUT / "figures/recency_gain")
    for k, v in numbers(rows, OUT / "numbers.tex").items():
        print(f"{k:18s} {v}")


if __name__ == "__main__":
    main()
