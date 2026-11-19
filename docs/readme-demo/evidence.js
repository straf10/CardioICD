// Evidence: paired bootstrap of the ensemble minus Greek BERT, then the McNemar split.
window.DURATION = 12;
const D = window.DATA;
const B = D.boot;
const SD = (B.hi - B.lo) / 3.92; // from the 95% interval; used only to draw the illustrative shape
const r = rng(7);
const SAMPLES = Array.from({ length: 2400 }, () => B.mean + SD * normal(r));
const [x0, x1, yb, hmax] = [96, 760, 470, 250];
const lo = -0.01;
const hi = 0.06;
const X = (v) => lerp(x0, x1, (v - lo) / (hi - lo));
const NB = 66;
const bins = new Array(NB).fill(0);
const binOf = (v) => clamp(Math.floor(((v - lo) / (hi - lo)) * NB), 0, NB - 1);
const upTo = new Array(SAMPLES.length); // bin counts are rebuilt per frame from the first n samples
SAMPLES.forEach((v, i) => (upTo[i] = binOf(v)));
const peak = (() => {
  const c = new Array(NB).fill(0);
  upTo.forEach((b) => c[b]++);
  return Math.max(...c);
})();

start((ctx, t) => {
  head(ctx, "Is the gain real?", "10,000 resampled test sets. In each one, ensemble micro-F1 minus Greek BERT micro-F1.");
  rrect(ctx, 56, 140, 740, 460, 16, "rgba(255,255,255,0.94)", C.rule);
  // axis
  ctx.strokeStyle = C.rule;
  ctx.beginPath();
  ctx.moveTo(x0, yb);
  ctx.lineTo(x1, yb);
  ctx.stroke();
  for (const v of [0, 0.02, 0.04, 0.06]) text(ctx, v ? "+" + v.toFixed(2) : "0", X(v), yb + 22, { size: 12, color: C.muted, align: "center" });
  text(ctx, "micro-F1 difference per resample", (x0 + x1) / 2, yb + 44, { size: 13, color: C.soft, align: "center" });
  // no-difference line
  ctx.strokeStyle = C.slate;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(X(0), 176);
  ctx.lineTo(X(0), yb);
  ctx.stroke();
  ctx.setLineDash([]);
  text(ctx, "no difference", X(0), 168, { size: 12, color: C.muted, align: "center" });

  // histogram grows as resamples arrive
  const n = Math.round(easeInOut(seg(t, 0.4, 5.0)) * SAMPLES.length);
  const c = new Array(NB).fill(0);
  for (let i = 0; i < n; i++) c[upTo[i]]++;
  const bw = (x1 - x0) / NB;
  c.forEach((v, i) => {
    const h = (v / peak) * hmax;
    ctx.fillStyle = C.ramp[3];
    ctx.fillRect(x0 + i * bw + 1, yb - h, bw - 2, h);
  });
  text(ctx, "illustrative shape: mean and 95% interval are the measured values", x0, 614, { size: 12, color: C.muted });
  text(ctx, `${(Math.round((n / SAMPLES.length) * B.n)).toLocaleString("en-US")} of ${B.n.toLocaleString("en-US")} resamples`, x1, 168, { size: 13, color: C.soft, align: "right" });

  // interval
  const ia = easeOut(seg(t, 5.2, 6.0));
  ctx.save();
  ctx.globalAlpha = ia;
  ctx.strokeStyle = C.accentDeep;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(X(B.lo), yb + 70 - 8);
  ctx.lineTo(X(B.hi), yb + 70 - 8);
  ctx.stroke();
  ctx.fillStyle = C.accentDeep;
  ctx.beginPath();
  sq(ctx, X(B.mean), yb + 62, 7);
  ctx.fill();
  ctx.restore();
  text(ctx, `mean +${B.mean.toFixed(4)}   95% interval [+${B.lo.toFixed(4)}, +${B.hi.toFixed(4)}]`, X(B.mean), yb + 82, { size: 14, weight: 650, color: C.accentDeep, align: "center", alpha: ia });

  // tiles on the right
  const s1 = easeOut(seg(t, 5.6, 6.2));
  rrect(ctx, 836, 150, 308, 130, 14, "#ffffff", C.accent, s1);
  text(ctx, "resamples that favour", 858, 184, { size: 14, color: C.soft, alpha: s1 });
  text(ctx, "Greek BERT alone", 858, 204, { size: 14, color: C.soft, alpha: s1 });
  text(ctx, `0 of ${B.n.toLocaleString("en-US")}`, 858, 256, { size: 36, weight: 700, color: C.accent, alpha: s1 });

  const s2 = easeOut(seg(t, 7.4, 8.0));
  const m = D.mcnemar;
  text(ctx, "Gold groups recovered by one side only (McNemar)", 836, 330, { size: 13, weight: 650, color: C.soft, alpha: s2 });
  [["Ensemble", m.ens, C.accent, 372], ["Greek BERT", m.bert, C.slate, 430]].forEach(([nm, v, col, y], i) => {
    const g = easeOut(seg(t, 7.6 + i * 0.5, 8.6 + i * 0.5));
    text(ctx, nm, 836, y, { size: 15, alpha: s2 });
    rrect(ctx, 836, y + 10, Math.max(8, 240 * (v / m.ens) * g), 18, 9, col, null, s2);
    text(ctx, String(Math.round(v * g)), 836 + 240 * (v / m.ens) * g + 12, y + 25, { size: 18, weight: 700, alpha: s2 });
  });
  text(ctx, "p ≈ 2.6 × 10⁻¹⁰", 836, 500, { size: 18, weight: 700, color: C.accentDeep, alpha: easeOut(seg(t, 9.6, 10.2)) });
  text(ctx, "Paired tests on the official test set.", 836, 526, { size: 13, color: C.muted, alpha: easeOut(seg(t, 9.6, 10.2)) });
});
