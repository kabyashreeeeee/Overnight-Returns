"""Generate the assignment research report."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import metrics as MET
from src.paths import submission_dir

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SUB = submission_dir(ROOT)

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceAfter=5, textColor=colors.HexColor("#1a1a1a"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=10.5, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=8.4, leading=11, alignment=TA_LEFT)
CELL = ParagraphStyle("Cell", parent=ss["BodyText"], fontSize=7.2, leading=8.6)
SMALL = ParagraphStyle("Small", parent=ss["BodyText"], fontSize=7.4, leading=9.4,
                       textColor=colors.HexColor("#444444"))


def tbl(rows, widths=None, align_right_from=1):
    def as_cell(value, header):
        if isinstance(value, Paragraph):
            return value
        return Paragraph(f"<b>{value}</b>" if header else str(value), CELL)

    data = [[as_cell(c, i == 0) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fb")]),
    ]))
    return t


def fig_image(fig, width=17 * cm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    img.drawWidth = width
    img.drawHeight = width * img.imageHeight / img.imageWidth
    return img


def main():
    stats = pd.read_csv(SUB / "statistics.csv")
    preds = pd.read_csv(SUB / "predictions.csv", parse_dates=["pred_date", "target_date"])
    acts = pd.read_csv(SUB / "actuals.csv", parse_dates=["pred_date", "target_date"])
    d = preds.merge(acts, on=["pred_date", "target_date", "symbol"])
    d["resid"] = d.actual_return_pct - d.universe_mean_pct
    sel = json.loads((OUT / "selection_manifest.json").read_text())
    integrity = pd.read_csv(OUT / "integrity_report.csv")
    magtab = pd.read_csv(OUT / "magnitude_validation.csv")
    conftab = pd.read_csv(OUT / "direction_confidence_validation.csv")
    mctab = pd.read_csv(OUT / "magnitude_confidence_validation.csv")
    lamtab = pd.read_csv(OUT / "direction_lambda_oof.csv")
    ablation = pd.read_csv(OUT / "direction_benchmark_ablation.csv")
    robustness = pd.read_csv(OUT / "economic_robustness.csv")

    def S(split, scope, metric):
        r = stats[(stats.split == split) & (stats.scope == scope) & (stats.metric == metric)]
        return float(r.value.iloc[0]) if len(r) else float("nan")

    def ds(dd, a):
        return float(np.sum(dd * a) / np.sum(np.abs(a)))

    story = []
    A = story.append

    # ---------------- page 1 ----------------
    A(Paragraph("Overnight Return Prediction — Quant Researcher Intern, Equity Desk", H1))
    A(Paragraph("Kabyashree Dey", H2))
    A(Paragraph("A four-output forecasting system for next-session gap magnitude, direction, "
                "direction correctness probability and magnitude reliability across 208 Indian "
                "equities, June 2020 – June 2026.", BODY))
    A(Spacer(1, 5))
    A(Paragraph("Headline results", H2))
    rows = [["Metric", "Train", "Validation", "Test"]]
    for lab, sc, m in [("Magnitude score", "pooled", "magnitude_score"),
                       ("Direction score (pooled)", "pooled", "direction_score"),
                       ("Direction score (residual)", "residual", "direction_score"),
                       ("r2 vs trailing-20 baseline", "pooled", "r2_vs_vol"),
                       ("Rank IC (magnitude)", "pooled", "rank_ic"),
                       ("Direction-confidence lift", "pooled", "conf_direction_lift"),
                       ("Brier skill", "pooled", "brier_skill"),
                       ("ECE-10", "pooled", "ece_10"),
                       ("Magnitude-confidence score", "pooled", "conf_magnitude_score")]:
        rows.append([lab] + [f"{S(s, sc, m):.4f}" for s in ["train", "valid", "test"]])
    A(tbl(rows, [6.2 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm]))
    A(Spacer(1, 5))

    te = d[d.split == "test"]
    up = np.ones(len(te))
    A(Paragraph(
        f"<b>Thesis.</b> The overnight return decomposes as a dominant common market gap, a weak "
        f"stock-specific residual, and noise. The four outputs are modelled separately because "
        f"they are statistically different problems: magnitude needs a non-negative conditional-mean "
        f"model, the market leg has ~935 training observations and needs heavy shrinkage, the residual "
        f"leg has ~182,000 and can carry a tree, and the two confidences are properties of the "
        f"<i>estimate</i> rather than of the return. Pooled "
        f"performance is largely market timing — an always-up book already scores "
        f"{ds(up, te.actual_return_pct.to_numpy()):.4f} on test — so the residual scope is where "
        f"stock selection actually shows. Residual direction score is "
        f"{S('test','residual','direction_score'):.4f} on test.", BODY))
    A(Spacer(1, 4))
    A(Paragraph(
        "<b>The one decision that shapes every number below.</b> The brief asks for "
        "E[|r| | F(T)], a conditional <i>mean</i>. The headline magnitude_score is an L1 statistic, "
        "minimised by the conditional <i>median</i>. These are different functionals (Gneiting 2011). "
        "We target the mean and accept the score cost. A semantic gate refuses to label any candidate "
        "E[|r|] unless its validation mean sits within 5% of realised and its calibration slope is in "
        "[0.7, 1.4]. The score-optimised quantile model is retained as a documented shadow: it scores "
        f"{magtab[magtab.model.str.startswith('S_')].magnitude_score.max():.4f} on validation but "
        "understates the mean by 28%, so it is not a conditional expectation and is not submitted.", BODY))

    rows = [["Coverage", "Rows", "Sessions", "Symbols"]]
    for s in ["train", "valid", "test"]:
        g = preds[preds.split == s]
        rows.append([s, f"{len(g):,}", g.pred_date.nunique(), g.symbol.nunique()])
    A(Spacer(1, 4)); A(tbl(rows, [4 * cm, 4 * cm, 4 * cm, 4 * cm]))
    A(Paragraph("Rows are omitted only where the next-session open does not exist or the 20-session "
                "feature burn-in is unmet; no name is dropped with hindsight.", SMALL))
    A(PageBreak())

    # ---------------- page 2: data + validation ----------------
    A(Paragraph("1. Data, target alignment and validation design", H1))
    A(Paragraph(
        "The target is 100 x (open(T+1)/close(T) - 1), where T+1 is the next session in the "
        "<b>master exchange calendar</b> built from the union of all symbol dates — never a symbol's "
        "own next available row. A halted stock therefore yields a missing target rather than silently "
        "borrowing a later session's open. Exact zeros map to +1.", BODY))
    keep = ["symbols", "sessions", "daily_rows", "duplicate_symbol_date", "negative_volume",
            "ohlc_bound_violations", "close_vs_1529_mean_bp", "close_vs_1529_mean_abs_bp"]
    rows = [["Integrity check", "Value"]]
    for _, r in integrity[integrity.check.isin(keep)].iterrows():
        rows.append([r.check.replace("_", " "), f"{r.value:,.4f}".rstrip("0").rstrip(".")])
    A(Spacer(1, 4)); A(tbl(rows, [9 * cm, 4 * cm]))
    A(Spacer(1, 4))
    A(Paragraph(
        "<b>The official close is not the last trade.</b> The daily close differs from the 15:29 minute "
        "close by 20.5 bp on average in absolute terms, but is within 2.9 bp of an approximate final-30-"
        "minute volume-weighted price constructed from the minute bars. The exact identity is "
        "1 + r_official = (1 + g_basis)(1 + r_1529_to_open), or r_official = g_basis + "
        "r_1529_to_open + g_basis x r_1529_to_open. The additive form is only a small-return "
        "approximation. The basis term is <i>observable at the close of T</i> and carries about 10% of "
        "the target's variance. We include "
        "it as a feature (bench_gap_pct) because it is legitimately inside F(T), and flag it in Section 8 "
        "as measurement structure rather than tradeable alpha.", BODY))
    A(Spacer(1, 4))
    rows = [["Block", "Prediction dates", "Rows"]]
    for s, lab in [("train", "Train"), ("valid", "Validation"), ("test", "Test")]:
        g = preds[preds.split == s]
        rows.append([lab, f"{g.pred_date.min().date()} to {g.pred_date.max().date()}", f"{len(g):,}"])
    A(tbl(rows, [3.5 * cm, 7 * cm, 3.5 * cm]))
    A(Paragraph(
        "Both embargoes are exactly five master-calendar sessions (2024-04-01..05 and 2025-05-02..08). "
        "Model selection uses five <b>expanding-window OOF folds built entirely inside training</b>: the "
        "final 657 training sessions are split into five contiguous blocks, each predicted by a model "
        "trained only on strictly earlier sessions with the same five-session embargo. No K-fold, no "
        "shuffling. Every fitted object — imputer, scaler, log-smearing factor, calibrator — is fit on "
        "the fold's own training rows.", BODY))
    A(Spacer(1, 3))
    A(Paragraph(
        "<b>Validation and test history.</b> " + sel["disclosure"] + " The fresh specification used no "
        "test rows to choose features, candidates, hyperparameters, blend weights or calibration: it was "
        "selected and frozen from training and validation first, evaluated on test afterward, and never "
        "changed after that result. Separately, an earlier package had already been evaluated on test. "
        "The later choice between the two packages was therefore test-informed, so the fresh test metrics "
        "are descriptive rather than pristine holdout selection evidence.", SMALL))
    A(Spacer(1, 8))

    # ---------------- page 3: features ----------------
    A(Paragraph("2. Minute data and feature hypotheses", H1))
    A(Paragraph(
        "Minute bars are filtered to 09:15–15:29 and aggregated into five-minute buckets anchored at "
        "09:15 (first open, max high, min low, last close, summed volume). Incomplete buckets are "
        "retained and counted rather than dropped, and coverage is carried as a feature because sparse "
        "trading should itself imply a less reliable forecast. 127 features in eight families:", BODY))
    rows = [["Family", "Hypothesis", "Construction", "Latest input"]]
    for f, h, c, t in [
        ("Overnight persistence", "Gap scale clusters across horizons",
         "Shifted lags, 5/20/60-day mean/median/std/EWM of |r|", "prior target"),
        ("Realised risk", "Volatility and its asymmetry forecast gap size",
         "Full/morning/final-30 RV, up/down semivariance, vol-of-vol", "15:29 bar of T"),
        ("Intraday path", "Closing pressure proxies unresolved order flow",
         "Final 60/30/15-min returns, trend slope, path efficiency", "15:29 bar of T"),
        ("Benchmark component", "Official close differs systematically from the last trade",
         "bench_gap_pct = close_1529/close_official - 1", "close of T"),
        ("Cross-section", "Separate common gap from stock-specific state",
         "Breadth, dispersion, market RV, ranks and z-scores within date", "all stocks to close T"),
        ("Stock-relative", "Rank in a calm market differs from rank in a crisis",
         "Raw level and rank and z-score and difference-from-mean, all kept", "close of T"),
        ("Liquidity", "Illiquidity widens the opening gap distribution",
         "Amihud, ADV, volume surprise, late-volume share", "close of T"),
        ("Data quality", "Sparse paths should imply lower reliability",
         "Minute coverage, incomplete buckets, history length flags", "close of T"),
    ]:
        rows.append([Paragraph(f"<b>{f}</b>", CELL), Paragraph(h, CELL),
                     Paragraph(c, CELL), Paragraph(t, CELL)])
    A(Spacer(1, 4)); A(tbl(rows, [3.2 * cm, 4.3 * cm, 6.3 * cm, 3.2 * cm]))
    A(Spacer(1, 4))
    A(Paragraph(
        "<b>Raw levels are kept alongside cross-sectional transforms.</b> A stock ranked most volatile "
        "on a calm day is not economically the same object as one ranked most volatile in a crisis, so "
        "discarding the level in favour of the rank throws away regime information.", BODY))
    A(Paragraph(
        "<b>|r| has a point mass at zero.</b> About 7.4% of training rows have exactly zero overnight "
        "return — the T+1 auction prints at the previous close, concentrated in illiquid names. That "
        "makes Tweedie with 1&lt;p&lt;2 (compound Poisson-Gamma) the theoretically correct family; Gamma "
        "requires a positive floor. Both are in the tournament.", BODY))
    A(PageBreak())

    # ---------------- page 4: magnitude ----------------
    A(Paragraph("3. Magnitude — a conditional mean, not a score-optimised quantile", H1))
    show = magtab[(magtab.variant == "raw") | (magtab.model == sel["selected"]["magnitude"])]
    show = show.sort_values("magnitude_score", ascending=False).head(9)
    rows = [["Candidate", "Variant", "Mag score", "MAE", "RMSE", "Rank IC", "r2 vs vol",
             "Mean gap %", "Slope", "Gate"]]
    for _, r in show.iterrows():
        rows.append([r.model.replace("_", " "), r.variant, f"{r.magnitude_score:.4f}",
                     f"{r.mae:.4f}", f"{r.rmse:.4f}", f"{r.rank_ic:.4f}", f"{r.r2_vs_vol:.4f}",
                     f"{r.mean_gap_pct:+.2f}", f"{r.calib_slope:.3f}",
                     "pass" if (r.passes_mean_gate and r.passes_slope_gate) else "FAIL"])
    A(tbl(rows, [3.3 * cm, 1.9 * cm, 1.7 * cm, 1.4 * cm, 1.4 * cm, 1.5 * cm, 1.5 * cm,
                 1.6 * cm, 1.3 * cm, 1.2 * cm]))
    A(Spacer(1, 4))
    A(Paragraph(
        f"The submitted forecast is a non-negative OOF-weighted blend of Gamma, Tweedie, Poisson, "
        f"log-L2-with-smearing and HAR-X Ridge mean models — every component a conditional-mean "
        f"estimator. Log-target models use Duan smearing rather than expm1, because E[exp(Z)] != "
        f"exp(E[Z]). Blend weights are fitted by non-negative least squares on OOF predictions only.", BODY))
    A(Paragraph(
        f"<b>The cost of correctness, quantified.</b> The score-optimised quantile shadow reaches "
        f"{magtab[magtab.model.str.startswith('S_')].magnitude_score.max():.4f} on validation but its "
        f"mean is 28% below realised — it is a median wearing a mean's label. The submitted blend scores "
        f"{S('valid','pooled','magnitude_score'):.4f} with a mean gap of "
        f"{sel['validation_scores']['magnitude_mean_gap_pct']:+.2f}%. We pay about 0.027 of headline score to "
        f"return the estimand the brief actually asked for, and gain on the stricter baseline-relative "
        f"measure: r2_vs_vol is {S('valid','pooled','r2_vs_vol'):.4f} on validation and "
        f"{S('test','pooled','r2_vs_vol'):.4f} on test, against 0.0000 for the trailing-20 baseline "
        f"by construction.", BODY))
    A(Spacer(1, 3))
    A(Paragraph(
        f"Magnitude skill decays as expected out of sample: score {S('train','pooled','magnitude_score'):.4f} "
        f"train, {S('valid','pooled','magnitude_score'):.4f} validation, {S('test','pooled','magnitude_score'):.4f} "
        f"test; rank IC {S('train','pooled','rank_ic'):.4f} / {S('valid','pooled','rank_ic'):.4f} / "
        f"{S('test','pooled','rank_ic'):.4f}. The gap's size and shape is the honest generalisation "
        f"diagnostic; no claim rests on the in-sample column.", BODY))
    A(Spacer(1, 8))

    # ---------------- page 5: direction ----------------
    A(Paragraph("4. Direction — market plus shrunk residual", H1))
    A(Paragraph(
        "Direction is sign(E[r | F(T)]) with E[r] = mu_market + lambda x mu_residual. The market leg is a "
        "Ridge on ~935 daily observations of the cross-sectional mean gap, with alpha chosen on a "
        "chronological inner holdout; anything more flexible would overfit that sample size. The residual "
        "leg is LightGBM under squared loss on the cross-sectionally demeaned target, so it estimates a "
        "conditional mean rather than a rank.", BODY))
    rows = [["lambda", "Pooled DS", "Residual DS", "Hit rate", "Fraction long"]]
    for _, r in lamtab.iterrows():
        rows.append([f"{r['lambda']:.2f}", f"{r.pooled_ds:.4f}", f"{r.residual_ds:.4f}",
                     f"{r.hit:.4f}", f"{r.frac_long:.4f}"])
    A(Spacer(1, 4)); A(tbl(rows, [2.6 * cm, 3.2 * cm, 3.2 * cm, 3 * cm, 3 * cm]))
    A(Paragraph(
        f"<b>lambda is selected on the inner OOF folds only — official validation is never used for it.</b> "
        f"It is a signal-strength parameter applied to the expected return before the sign is taken, not a "
        f"post-hoc demeaning of an emitted score. The selected value is "
        f"{sel['selected']['lambda']:.2f}.", BODY))
    A(Spacer(1, 3))
    tev = d[d.split == "test"]
    A(Paragraph(
        f"<b>Pooled versus residual.</b> On test the market leg alone reproduces the always-up book "
        f"almost exactly — with a +{100 * tev.actual_return_pct.mean():.2f} bp/night average gap it never "
        f"predicts a down day — so pooled "
        f"direction score {S('test','pooled','direction_score'):.4f} is mostly market timing. The model is "
        f"built to capture <i>both</i>, but the incremental evidence is in the residual scope: residual "
        f"direction score is {S('test','residual','direction_score'):.4f} on test and "
        f"{S('valid','residual','direction_score'):.4f} on validation, against 0.0000 for any book that "
        f"ignores the cross-section. Residual hit rate is {S('test','residual','hit_rate'):.4f} — barely "
        f"above a coin flip — so the residual edge is <i>value-weighted</i>: the model gets the large "
        f"stock-specific moves right and loses on small ones.", BODY))
    A(Spacer(1, 3))
    A(Paragraph(
        f"Pooled hit rate is {S('test','pooled','hit_rate'):.4f} on test against "
        f"{(tev.actual_return_pct >= 0).mean():.4f} for always-up. The model is behind on raw sign counting "
        f"and ahead on the value-weighted score — an honest description of what it does.", BODY))
    A(PageBreak())

    # ---------------- page 6: confidences ----------------
    A(Paragraph("5. Direction confidence", H1))
    rows = [["Candidate", "Brier", "Brier skill", "Log loss", "ECE-10", "Lift", "Flips"]]
    for _, r in conftab.iterrows():
        rows.append([r.model.replace("_", " "), f"{r.brier:.4f}", f"{r.brier_skill:.4f}",
                     f"{r.log_loss:.4f}", f"{r.ece_10:.4f}", f"{r.lift:+.4f}", int(r.flips)])
    A(tbl(rows, [4.6 * cm, 2 * cm, 2.2 * cm, 2 * cm, 2 * cm, 2 * cm, 1.6 * cm]))
    A(Spacer(1, 4))
    A(Paragraph(
        "The label is correctness of the <i>emitted</i> direction, on genuine OOF rows. The selected model "
        "combines two <b>independently fitted</b> objects: the mean model's aligned margin and the quantile "
        "model's conditional CDF at zero. That is not a rescaling of the direction score — the two disagree "
        "on about 13% of rows, because sign(E[r]) and sign(median) diverge under skew. Candidates must clear "
        "positive lift and non-negative Brier skill before being ranked on Brier.", BODY))
    A(Paragraph(
        f"On test: Brier skill {S('test','pooled','brier_skill'):.4f}, ECE-10 "
        f"{S('test','pooled','ece_10'):.4f}, lift {S('test','pooled','conf_direction_lift'):+.4f}. "
        f"If P(correct) &lt; 0.5 the direction is flipped and confidence becomes 1-p, so conf_direction is "
        f"never self-contradictory (minimum {preds.conf_direction.min():.4f}).", BODY))
    A(Paragraph(
        f"Train ECE-10 is higher ({S('train','pooled','ece_10'):.4f}) than validation "
        f"({S('valid','pooled','ece_10'):.4f}) or test ({S('test','pooled','ece_10'):.4f}) because the "
        "reported train directions come from in-sample final fits while the confidence calibrator learned "
        "correctness from chronological OOF predictions. The resulting train under-confidence is an "
        "interface mismatch, not evidence that validation or test calibrators saw their labels.", SMALL))
    A(Spacer(1, 4))
    A(Paragraph(
        f"<b>Residual-scope calibration is deliberately not claimed.</b> conf_direction estimates "
        f"P(emitted sign matches the <i>raw</i> return). Residual correctness is a different question, and "
        f"the residual rows show it: Brier skill {S('test','residual','brier_skill'):.4f}, ECE "
        f"{S('test','residual','ece_10'):.4f} on test. This is an estimand limitation, stated rather than "
        f"hidden — the number should not be read as a stock-selection probability.", BODY))

    A(Spacer(1, 6)); A(Paragraph("6. Magnitude confidence — splitting scale from reliability", H1))
    rows = [["Candidate", "Score", "Within-mag-decile", "Corr with magnitude", "Top MAE", "Bottom MAE"]]
    for _, r in mctab.iterrows():
        rows.append([r.model.replace("_", " "), f"{r.conf_magnitude_score:.4f}",
                     f"{r.within_magnitude_decile:.4f}", f"{r.corr_with_magnitude:.3f}",
                     f"{r.mae_top_decile:.4f}", f"{r.mae_bottom_decile:.4f}"])
    A(tbl(rows, [4.2 * cm, 2.2 * cm, 3.4 * cm, 3.4 * cm, 2 * cm, 2.2 * cm]))
    A(Paragraph(
        "Raw |m - |a|| is dominated by aleatoric variance, so any model of it collapses onto "
        "-pred_magnitude: the naive baseline already scores 0.328. Stage 1 absorbs that scale term with an "
        "isotonic fit of E[error | m]; stage 2 predicts the scale-free remainder from interval width, error "
        "history, coverage and regime. The <b>within-magnitude-decile</b> column is the non-degeneracy test — "
        "it holds scale roughly fixed, and the selected model nearly doubles the baseline there "
        f"({mctab.within_magnitude_decile.max():.4f} versus "
        f"{mctab[mctab.model=='D0_neg_magnitude'].within_magnitude_decile.iloc[0]:.4f}).", BODY))
    A(PageBreak())

    # ---------------- page 7: breadth, stability, costs ----------------
    A(Paragraph("7. Breadth, stability and transaction costs", H1))
    by = te.groupby("symbol").apply(lambda x: pd.Series({
        "n": len(x), "hit": (x.pred_direction == np.where(x.actual_return_pct >= 0, 1, -1)).mean(),
        "naive": (x.actual_return_pct >= 0).mean()}), include_groups=False)
    by = by[by.n >= 20]; by["edge"] = by.hit - by.naive
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.1))
    axes[0].hist(by.hit, bins=25, color="#4C78A8", alpha=.85)
    axes[0].axvline(0.5, color="#888", ls="--"); axes[0].set_title("Test hit rate by symbol", fontsize=9)
    axes[0].set_xlabel("hit rate", fontsize=8)
    mo = d[d.split == "test"].groupby(d[d.split == "test"].pred_date.dt.to_period("M")).apply(
        lambda x: pd.Series({"pooled": ds(x.pred_direction.to_numpy(), x.actual_return_pct.to_numpy()),
                             "residual": ds(x.pred_direction.to_numpy(), x.resid.to_numpy())}),
        include_groups=False)
    axes[1].plot(range(len(mo)), mo.pooled, marker="o", ms=3, label="pooled")
    axes[1].plot(range(len(mo)), mo.residual, marker="o", ms=3, label="residual")
    axes[1].axhline(0, color="#888", lw=.8)
    axes[1].set_xticks(range(0, len(mo), 3)); axes[1].set_xticklabels([str(p) for p in mo.index[::3]], fontsize=6, rotation=45)
    axes[1].set_title("Test direction score by month", fontsize=9); axes[1].legend(fontsize=7)
    A(fig_image(fig))
    A(Spacer(1, 3))
    quarterly = te.groupby(te.pred_date.dt.to_period("Q")).apply(
        lambda x: pd.Series({
            "sessions": x.pred_date.nunique(),
            "pooled": ds(x.pred_direction.to_numpy(), x.actual_return_pct.to_numpy()),
            "residual": ds(x.pred_direction.to_numpy(), x.resid.to_numpy()),
            "magnitude": 1.0 - np.abs(
                x.pred_magnitude_pct.to_numpy() - x.actual_magnitude_pct.to_numpy()
            ).sum() / x.actual_magnitude_pct.sum(),
        }), include_groups=False)
    A(Paragraph("Quarterly out-of-sample stability", H2))
    rows = [["Test quarter", "Sessions", "Pooled DS", "Residual DS", "Magnitude score"]]
    for q, r in quarterly.iterrows():
        rows.append([str(q), int(r.sessions), f"{r.pooled:.4f}",
                     f"{r.residual:.4f}", f"{r.magnitude:.4f}"])
    A(tbl(rows, [3.3 * cm, 2.4 * cm, 3.2 * cm, 3.2 * cm, 3.6 * cm]))
    A(Paragraph(
        f"All {len(quarterly)} test-quarter segments have positive pooled and residual direction scores, "
        f"but the range is wide: pooled {quarterly.pooled.min():.4f}–{quarterly.pooled.max():.4f}, "
        f"residual {quarterly.residual.min():.4f}–{quarterly.residual.max():.4f}. The latest quarter is "
        f"weaker on residual score ({quarterly.iloc[-1].residual:.4f}), so stability means persistent sign, "
        "not constant strength. Quarterly magnitude scores also vary sharply because the headline is a "
        "ratio of pooled absolute-error sums, not an equal-weighted average of quarter scores: quarters "
        "with more observations and larger realised |r| carry more denominator weight. The first test "
        "quarter is partial because the embargo ends in May.", SMALL))
    A(Spacer(1, 3))
    best10 = by.edge.nlargest(10); worst10 = by.edge.nsmallest(10)
    rows = [["#", "Best (test, edge over own always-up)", "Edge", "Worst", "Edge"]]
    for i, ((bs, bv), (ws, wv)) in enumerate(zip(best10.items(), worst10.items()), 1):
        rows.append([i, bs, f"{bv:+.3f}", ws, f"{wv:+.3f}"])
    A(tbl(rows, [1 * cm, 6 * cm, 2.2 * cm, 5 * cm, 2.2 * cm]))
    va = d[d.split == "valid"]
    by_v = va.groupby("symbol").apply(lambda x: pd.Series({
        "n": len(x), "hit": (x.pred_direction == np.where(x.actual_return_pct >= 0, 1, -1)).mean(),
        "naive": (x.actual_return_pct >= 0).mean()}), include_groups=False)
    by_v = by_v[by_v.n >= 20]; by_v["edge"] = by_v.hit - by_v.naive
    joined = by_v[["edge"]].join(by[["edge"]], lsuffix="_v", rsuffix="_t", how="inner").dropna()
    pers = spearmanr(joined.edge_v, joined.edge_t)
    A(Paragraph(
        f"All {(by.hit > 0.5).mean():.0%} of symbols clear a 50% hit rate on test, but that mostly "
        f"reflects positive drift; only {S('test','pooled','frac_stocks_beat_naive'):.1%} beat their own "
        f"always-up rate, versus {S('valid','pooled','frac_stocks_beat_naive'):.1%} on validation. Ranking "
        f"symbols by edge is more informative than by hit rate. <b>The winning subset is not identifiable in "
        f"advance:</b> the rank correlation between a symbol's validation edge and its subsequent test edge "
        f"is {pers.statistic:+.3f} (p = {pers.pvalue:.2f}, n = {len(joined)}) — indistinguishable from zero. "
        f"We therefore apply no ex-post universe filter. The positive value-weighted residual score alongside "
        f"only {S('test','pooled','frac_stocks_beat_naive'):.1%} of stocks beating their own naive rate means "
        f"the edge is concentrated in a minority of names on larger moves, not broad stock-level superiority.", BODY))
    A(Spacer(1, 3))
    gross = 100 * np.mean(te.pred_direction.to_numpy() * te.actual_return_pct.to_numpy())
    drift = 100 * te.actual_return_pct.mean()
    resid_bp = 100 * np.mean(te.pred_direction.to_numpy() * te.resid.to_numpy())
    A(Paragraph(
        f"<b>Transaction costs.</b> The trade is a full round trip every night: establish at the close of T, "
        f"liquidate in the T+1 auction, so turnover is 100% per night regardless of whether the sign flips. "
        f"Gross directional return is {gross:.2f} bp/night on test, of which {drift:.2f} bp is common drift "
        f"available from always-up; the incremental edge is {gross-drift:.2f} bp and the residual long/short "
        f"book {resid_bp:.2f} bp. A 60-stock train/validation diagnostic gives a median effective spread "
        f"of about 5.5 bp (Roll 1984 on 1-minute 10:00–15:00 returns); a rough Amihud calculation gives "
        f"2–5 bp of impact at 5% ADV. Cash-delivery STT is 10 bp on purchase and 10 bp on sale, or 20 bp "
        f"round trip, before exchange fees and brokerage. <b>STT alone exceeds every incremental edge above.</b> "
        f"The 09:15 minute has a median "
        f"71.5 bp high-low range against 9.3 bp at midday — the exit sits in the worst liquidity window of "
        f"the day. Naked cash shorting is prohibited; an overnight short leg requires securities borrowing, "
        f"with its own availability and borrow cost. Single-stock futures permit shorting, but futures STT "
        f"is 5 bp on the sell side from 1 April 2026, and futures basis, roll and point-in-time eligibility "
        f"would require a separate target and study. "
        f"<b>We claim a statistical edge, not a tradeable one.</b>", BODY))
    A(PageBreak())

    # ---------------- page 8: robustness, ablation, uncertainty ----------------
    A(Paragraph("8. Economic robustness and uncertainty", H1))
    A(Paragraph(
        "The graded target uses the supplied official close. Two sensitivity targets separate "
        "assignment performance from executable-timestamp economics: the 15:29 last trade to the next "
        "open, and that last trade to the next close. These diagnostics were run after model selection "
        "and did not alter the submitted specification.", BODY))
    rt = robustness[robustness.split == "test"]
    rows = [["Target", "Pooled DS", "Residual DS", "Directional bp"]]
    labels = {
        "official_close": "Official close → next open (graded)",
        "last_trade_1529": "15:29 last trade → next open",
        "close_close": "15:29 last trade → next close",
    }
    for target, label in labels.items():
        pooled = rt[(rt.target == target) & (rt.scope == "pooled")].iloc[0]
        residual = rt[(rt.target == target) & (rt.scope == "residual")].iloc[0]
        rows.append([label, f"{pooled.direction_score:+.4f}",
                     f"{residual.direction_score:+.4f}",
                     f"{pooled.directional_return_bps:+.2f}"])
    A(Spacer(1, 4)); A(tbl(rows, [7.5 * cm, 3 * cm, 3 * cm, 3 * cm]))
    official_resid = float(rt[(rt.target == "official_close") & (rt.scope == "residual")].direction_score.iloc[0])
    last_resid = float(rt[(rt.target == "last_trade_1529") & (rt.scope == "residual")].direction_score.iloc[0])
    A(Paragraph(
        f"The residual score is {official_resid:+.4f} on the required target but {last_resid:+.4f} from "
        "the final traded price. <b>The incremental result therefore does not survive the executable-"
        "timestamp reconstruction.</b> The benchmark-basis feature is valid F(T) information for the "
        "assignment, but the strongest score should be read as target-definition structure rather than "
        "post-close alpha.", BODY))

    A(Spacer(1, 5)); A(Paragraph(
        "Validation-only benchmark-feature ablation of the base direction model", H2))
    A(Paragraph(
        "These rows use sign(mu_market + lambda x mu_residual) <b>before</b> the frozen Part 3 "
        "confidence-flip rule. They therefore differ from the headline validation scores (0.3220 pooled, "
        "0.0773 residual), which use the final emitted directions. Full and ablated specifications are "
        "compared on the same pre-confidence basis.", SMALL))
    rows = [["Specification", "Pooled DS", "Residual DS", "Residual rank IC",
             "Stocks beat naive"]]
    for _, r in ablation.iterrows():
        rows.append([r.model.replace("_", " "), f"{r.pooled_direction_score:.4f}",
                     f"{r.residual_direction_score:.4f}", f"{r.residual_rank_ic:.4f}",
                     f"{r.frac_stocks_beat_naive:.1%}"])
    A(tbl(rows, [6.5 * cm, 2.4 * cm, 2.5 * cm, 3 * cm, 2.6 * cm]))
    arow = ablation.iloc[0]
    A(Paragraph(
        f"At the frozen lambda, removing only the explicit benchmark-basis variables changes residual "
        f"direction score by {arow.full_minus_ablated_residual_mean:+.4f}; paired five-session 95% interval "
        f"[{arow.full_minus_ablated_residual_ci_low:+.4f}, {arow.full_minus_ablated_residual_ci_high:+.4f}]. "
        "This is a post-final validation diagnostic, introduced after the historical test was known; it "
        "does not alter the submitted model or claim a new clean selection pass.", BODY))
    A(Paragraph(
        "The modest explicit-feature ablation and the negative executable-target result answer different "
        "questions. The ablation removes only the named basis columns while leaving correlated late-session "
        "price and path features that can proxy the close-to-last-trade basis; it measures marginal dependence "
        "conditional on those proxies. Replacing the target removes that basis from the label itself. Because "
        "the score fails under the latter test, the conservative interpretation is that most of the residual "
        "result is target/measurement structure rather than executable alpha.", SMALL))

    A(Spacer(1, 5)); A(Paragraph("Paired block-bootstrap uncertainty", H2))
    valid_diag = MET.add_vol_baseline(d.copy())
    valid_diag = valid_diag[valid_diag.split == "valid"].sort_values(["pred_date", "symbol"]).reset_index(drop=True)
    va_resid = valid_diag.actual_return_pct.to_numpy() - valid_diag.universe_mean_pct.to_numpy()
    _, resid_lo, resid_hi = MET.block_bootstrap_diff(
        valid_diag, valid_diag.pred_direction.to_numpy(float), np.zeros(len(valid_diag)), va_resid,
        block_sessions=int(sel["config"]["bootstrap"]["block_sessions"]),
        draws=int(sel["config"]["bootstrap"]["draws"]), seed=42,
    )
    _, pool_lo, pool_hi = MET.block_bootstrap_diff(
        valid_diag, valid_diag.pred_direction.to_numpy(float), np.ones(len(valid_diag)),
        valid_diag.actual_return_pct.to_numpy(float),
        block_sessions=int(sel["config"]["bootstrap"]["block_sessions"]),
        draws=int(sel["config"]["bootstrap"]["draws"]), seed=43,
    )
    finite = valid_diag.vol_baseline_20.notna().to_numpy()
    vm = valid_diag.loc[finite].reset_index(drop=True)
    _, mag_lo, mag_hi = MET.block_bootstrap_magnitude_diff(
        vm, vm.pred_magnitude_pct.to_numpy(), vm.vol_baseline_20.to_numpy(),
        vm.actual_magnitude_pct.to_numpy(),
        block_sessions=int(sel["config"]["bootstrap"]["block_sessions"]),
        draws=int(sel["config"]["bootstrap"]["draws"]), seed=44,
    )
    verr = np.abs(valid_diag.pred_magnitude_pct.to_numpy() - valid_diag.actual_magnitude_pct.to_numpy())
    _, rel_lo, rel_hi = MET.block_bootstrap_spearman_diff(
        valid_diag, valid_diag.conf_magnitude.to_numpy(), -valid_diag.pred_magnitude_pct.to_numpy(), -verr,
        block_sessions=int(sel["config"]["bootstrap"]["block_sessions"]),
        draws=int(sel["config"]["bootstrap"]["draws"]), seed=45,
    )
    _, test_resid_lo, test_resid_hi = MET.block_bootstrap_diff(
        te.sort_values(["pred_date", "symbol"]).reset_index(drop=True),
        te.sort_values(["pred_date", "symbol"]).pred_direction.to_numpy(float),
        np.zeros(len(te)),
        te.sort_values(["pred_date", "symbol"]).resid.to_numpy(float),
        block_sessions=int(sel["config"]["bootstrap"]["block_sessions"]),
        draws=int(sel["config"]["bootstrap"]["draws"]), seed=52,
    )
    _, test_pool_lo, test_pool_hi = MET.block_bootstrap_diff(
        te.sort_values(["pred_date", "symbol"]).reset_index(drop=True),
        te.sort_values(["pred_date", "symbol"]).pred_direction.to_numpy(float),
        np.ones(len(te)),
        te.sort_values(["pred_date", "symbol"]).actual_return_pct.to_numpy(float),
        block_sessions=int(sel["config"]["bootstrap"]["block_sessions"]),
        draws=int(sel["config"]["bootstrap"]["draws"]), seed=53,
    )
    rows = [
        ["Comparison", "95% paired five-session interval"],
        ["Residual direction score versus zero", f"[{resid_lo:+.4f}, {resid_hi:+.4f}]"],
        ["Pooled direction score versus always-up", f"[{pool_lo:+.4f}, {pool_hi:+.4f}]"],
        ["Magnitude score versus trailing-20", f"[{mag_lo:+.4f}, {mag_hi:+.4f}]"],
        ["Magnitude confidence versus -m", f"[{rel_lo:+.4f}, {rel_hi:+.4f}]"],
        ["Test residual direction score versus zero", f"[{test_resid_lo:+.4f}, {test_resid_hi:+.4f}]"],
        ["Test pooled direction score versus always-up", f"[{test_pool_lo:+.4f}, {test_pool_hi:+.4f}]"],
    ]
    A(tbl(rows, [9 * cm, 7.5 * cm]))
    A(Paragraph(
        "The test residual interval excludes zero; the pooled improvement over always-up does not. Thus the "
        "pooled headline is not statistically distinguishable from passive positive overnight drift, while "
        "the residual statistic is positive but subject to the executable-target caveat above.", SMALL))

    test_mag = d[d.split == "test"].copy()
    test_err = np.abs(test_mag.pred_magnitude_pct - test_mag.actual_magnitude_pct)
    dec = pd.qcut(test_mag.pred_magnitude_pct.rank(method="first"), 10, labels=False)
    within_conf = np.nanmean([
        spearmanr(test_mag.loc[dec == k, "conf_magnitude"], -test_err[dec == k]).statistic
        for k in range(10)
    ])
    within_base = np.nanmean([
        spearmanr(-test_mag.loc[dec == k, "pred_magnitude_pct"], -test_err[dec == k]).statistic
        for k in range(10)
    ])
    overall_conf = spearmanr(test_mag.conf_magnitude, -test_err).statistic
    overall_base = spearmanr(-test_mag.pred_magnitude_pct, -test_err).statistic
    A(Paragraph(
        f"<b>Part 4 did not add robust information beyond scale on test.</b> Headline confidence score "
        f"{overall_conf:.4f} is below the simple -pred_magnitude baseline {overall_base:.4f}; within "
        f"magnitude deciles it is {within_conf:.4f} versus {within_base:.4f}. The two-stage hypothesis "
        "improved validation but did not generalise, so it is retained as the frozen output and explicitly "
        "demoted rather than redesigned after seeing test.", BODY))
    A(PageBreak())

    # ---------------- final page: negatives + limitations ----------------
    A(Paragraph("9. What did not work, and limitations", H1))
    rows = [["Attempt", "Evidence and decision"]]
    for a_, b_ in [
        ("Score-optimised quantile magnitude",
         f"Best validation magnitude score ({magtab[magtab.model.str.startswith('S_')].magnitude_score.max():.4f}) "
         "but understates the mean by 28%. Fails the semantic gate; kept as a shadow, not submitted."),
        ("Direct L2 on |r|",
         "The most literal conditional-mean estimator, but overshoots the mean by +3.4% and scores below "
         "the blend. Retained as a blend component rather than standalone."),
        ("Quantile P(r>0) alone as confidence",
         "Lift +0.12 but ECE 0.054 and 7,188 flips that cut pooled score to 0.262. Misaligned with an "
         "emitted direction taken from sign(E[r]). Rejected in favour of the combined model."),
        ("Single-stage magnitude reliability",
         "Scale-only and raw-error models both collapse to corr -0.999 with pred_magnitude and add nothing "
         "over the naive -m baseline. Motivated the two-stage split."),
        ("Two-stage magnitude reliability on test",
         "Improved the validation within-decile diagnostic, but did not beat -pred_magnitude on the frozen "
         "test block. Retained for protocol fidelity and explicitly demoted; not redesigned post-test."),
        ("Corwin-Schultz / Abdi-Ranaldo spreads",
         "Return ~130 bp for NSE large caps, implausible. Daily high-low estimators conflate volatility with "
         "spread in high-vol regimes; replaced with Roll on 1-minute returns (~5.5 bp)."),
        ("Higher-capacity magnitude trees",
         "900 estimators / 63 leaves gave no material validation gain over 600/31 at several times the "
         "runtime. Kept the smaller configuration."),
    ]:
        rows.append([Paragraph(f"<b>{a_}</b>", CELL), Paragraph(b_, CELL)])
    A(tbl(rows, [5 * cm, 12 * cm]))
    A(Paragraph("Limitations", H2))
    A(Paragraph(
        "The universe is the supplied file set, so survivorship is implicit and no point-in-time sector or "
        "event data is used. The residual statistic is positive but small, is largely tied to target-definition "
        "structure, and does not survive costs. conf_direction is "
        "calibrated for pooled correctness only. Quantile crossings before rearrangement rise from 4.8% on "
        "the development block to 17.5% on test, which is a genuine sign of distributional drift and a caveat "
        "on the confidence numbers. Validation was inspected twice for model development and later again for "
        "the explicitly post-final benchmark ablation. The overall-project test-history exception is disclosed "
        "once in Section 1; it cannot be retroactively converted into pristine holdout evidence. An always-up book "
        "scores well above zero on this data — the brief's "
        "claim that it scores 0 by construction holds only when the mean overnight return is zero, which it "
        "is not here (+21 bp/night), so always-up is reported as an explicit comparator throughout.", BODY))
    A(Spacer(1, 4))
    A(Paragraph(
        "<b>Reproduction.</b> Seed 42, fixed model threads, pinned dependencies and config-relative paths are "
        "used throughout. The documented command rebuilds the cache when absent and reproduces the three CSVs; "
        "full operational details and checks are in README.md and the packaged tests.", SMALL))
    A(Spacer(1, 4))
    A(Paragraph(
        "<b>References.</b> Gneiting (2011), <i>Making and Evaluating Point Forecasts</i>; Duan (1983), "
        "smearing estimation; Roll (1984), effective spreads; Corwin and Schultz (2012), bid-ask spreads "
        "from high/low prices; NSE Finance &amp; Accounts Circular 02/2026, securities-transaction-tax rates; "
        "SEBI master circular, short-selling and securities-lending framework. NSE closing-price methodology "
        "motivates, but does not make executable, the "
        "official-close benchmark component.", SMALL))

    doc = SimpleDocTemplate(str(SUB / "research.pdf"), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="Overnight Return Prediction",
                            author="Kabyashree Dey")
    doc.build(story)
    import pypdf
    n = len(pypdf.PdfReader(str(SUB / "research.pdf")).pages)
    print(f"wrote {SUB/'research.pdf'} ({n} pages)")


if __name__ == "__main__":
    main()
