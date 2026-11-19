// Bands: rule-based parts win on rare codes, Greek BERT wins on frequent codes.
window.DURATION = 12;
const D = window.DATA;
const rare = Object.entries(D.bands.rare).sort((a, b) => b[1] - a[1]);
const freq = Object.entries(D.bands.freq).sort((a, b) => b[1] - a[1]);

function panel(ctx, k, x, title, sub, rows, at, topName, a) {
  rrect(ctx, x, 150, 508, 380, 16, "rgba(255,255,255,0.94)", C.rule, a);
  text(ctx, title, x + 28, 192, { size: 20, weight: 650, alpha: a });
  text(ctx, sub, x + 28, 216, { size: 13, color: C.muted, alpha: a });
  const [bx0, bx1] = [x + 170, x + 440];
  rows.forEach(([nm, v], i) => {
    const y = 262 + i * 52;
    const s = easeOut(seg(k, at + i * 0.3, at + 0.9 + i * 0.3));
    const top = nm === topName;
    text(ctx, nm, x + 28, y + 6, { size: 16, weight: top ? 650 : 500, alpha: a * Math.min(1, s * 3) });
    rrect(ctx, bx0, y - 9, bx1 - bx0, 18, 9, C.accentSoft, null, a * Math.min(1, s * 3) * 0.55);
    rrect(ctx, bx0, y - 9, Math.max(18, (bx1 - bx0) * v * s), 18, 9, nm === "Ensemble" ? C.slate : top ? C.accent : C.ramp[2], null, a);
    text(ctx, (v * s).toFixed(2), bx1 + 12, y + 6, { size: 16, weight: 700, color: C.ink, alpha: a * Math.min(1, s * 3) });
  });
}

start((ctx, t) => {
  head(ctx, "Where the models win", "Band-level micro-F1, split by how often a code occurs in the corpus.");
  panel(ctx, t, 56, "Rare codes", "30 codes with fewer than 10 instances", rare, 0.6, "Dictionary", 1);
  panel(ctx, t, 636, "Frequent codes", "codes with 100 to 499 instances", freq, 5.2, "Greek BERT", seg(t, 4.8, 5.4));
  const c = easeOut(seg(t, 3.6, 4.2));
  text(ctx, "The ensemble is tuned for overall micro-F1 and does not recover the long tail.", 56, 580, { size: 16, color: C.soft, alpha: c * (1 - seg(t, 4.6, 5.0)) });
  const d = easeOut(seg(t, 8.0, 8.6));
  text(ctx, "The order flips: rules for the rare, Greek BERT for the frequent.", 56, 580, { size: 16, weight: 650, color: C.accentDeep, alpha: d });
  text(ctx, "Values from the paper analysis; ensemble ≈ 0.39 on the rare band.", 56, 614, { size: 13, color: C.muted, alpha: d });
});
