// Static card: official test set results. Rendered once as a PNG.
window.DURATION = 1;
const D = window.DATA;
const H2 = 540;

start((ctx) => {
  head(ctx, "Official test set results", "BioASQ / CLEF ELCardioCC 2026. Group-level micro-F1.");
  const best = D.test[0];
  [["F1", best[3], C.accent], ["Precision", best[2], C.accentDeep], ["Recall", best[1], C.accentDeep]].forEach(([lab, v, col], i) => {
    const x = 56 + i * 372;
    rrect(ctx, x, 130, 340, 100, 16, "#ffffff", i === 0 ? C.accent : C.rule);
    text(ctx, lab, x + 24, 164, { size: 15, color: C.soft });
    text(ctx, v.toFixed(4), x + 24, 212, { size: 40, weight: 700, color: col });
  });
  text(ctx, "Best submission: ensemble, merge_and", 56, 262, { size: 14, color: C.muted });

  // submissions: lollipop on F1
  const [x0, x1] = [420, 800];
  const X = (v) => lerp(x0, x1, (v - 0.84) / 0.03);
  text(ctx, "submission", 56, 304, { size: 13, weight: 650, color: C.soft });
  text(ctx, "micro-F1", x0, 304, { size: 13, weight: 650, color: C.soft });
  text(ctx, "precision", 960, 304, { size: 13, weight: 650, color: C.soft, align: "right" });
  text(ctx, "recall", 1060, 304, { size: 13, weight: 650, color: C.soft, align: "right" });
  D.test.forEach(([nm, rc, pr, f], i) => {
    const y = 338 + i * 36;
    const top = i === 0;
    text(ctx, nm, 56, y + 5, { size: 16, weight: top ? 650 : 500 });
    ctx.strokeStyle = C.accentSoft;
    ctx.lineWidth = 7;
    ctx.lineCap = "butt";
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(X(f), y);
    ctx.stroke();
    ctx.fillStyle = top ? C.accent : nm.startsWith("Greek") ? C.slate : C.ramp[2];
    ctx.beginPath();
    sq(ctx, X(f), y, 7);
    ctx.fill();
    text(ctx, f.toFixed(4), X(f) + 16, y + 5, { size: 15, weight: 700 });
    text(ctx, pr.toFixed(4), 960, y + 5, { size: 15, color: C.soft, align: "right" });
    text(ctx, rc.toFixed(4), 1060, y + 5, { size: 15, color: C.soft, align: "right" });
  });
  text(ctx, "Axis starts at 0.84. Gain over the best standalone model: +1.8 points (paired bootstrap, 10,000 resamples).", 56, 520, { size: 12, color: C.muted });
}, 1200, H2);
