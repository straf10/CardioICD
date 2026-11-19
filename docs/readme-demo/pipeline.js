// Pipeline: five steps, each with one small picture built from real sizes.
window.DURATION = 15;
const D = window.DATA;
const STEPS = ["Clean and split", "Train six models", "Score every note", "Search fusions", "Propose codes"];
const SEC = 3;
const FADE = 0.35;
const Z = D.sizes;

function chips(ctx, t) {
  STEPS.forEach((s, i) => {
    const on = t >= i * SEC && t < (i + 1) * SEC;
    const done = t >= (i + 1) * SEC;
    const w = 206;
    const x = 56 + i * (w + 14);
    rrect(ctx, x, 128, w, 40, 20, on ? C.accent : done ? C.accentSoft : "#ffffff", on ? null : C.rule);
    text(ctx, `${i + 1}  ${s}`, x + w / 2, 154, { size: 15, weight: 650, color: on ? "#fff" : done ? C.accentDeep : C.muted, align: "center" });
  });
}

function step1(ctx, k, a) {
  text(ctx, `${Z.docs.toLocaleString("en-US")} discharge summaries, ${Z.codes} target codes`, 76, 230, { size: 22, weight: 650, alpha: a });
  text(ctx, "Free-text Greek notes are cleaned, then split so that rare codes appear in training.", 76, 258, { size: 15, color: C.soft, alpha: a });
  const [x0, w] = [76, 1050];
  const parts = [["train", Z.train, C.accent], ["validation", Z.val, C.ramp[2]], ["test", Z.test, C.ramp[1]]];
  let x = x0;
  const grow = easeOut(seg(k, 0.3, 1.6));
  parts.forEach(([nm, n, col]) => {
    const pw = (w * n) / Z.docs * grow;
    rrect(ctx, x, 320, Math.max(pw - 4, 2), 70, 10, col, null, a);
    if (grow > 0.9) {
      text(ctx, n.toLocaleString("en-US"), x + (pw - 4) / 2, 362, { size: 24, weight: 700, color: nm === "train" ? "#fff" : C.accentDeep, align: "center", alpha: a });
      text(ctx, nm, x + (pw - 4) / 2, 416, { size: 14, color: C.soft, align: "center", alpha: a });
    }
    x += pw;
  });
  text(ctx, "multi-label stratified split, 80 / 10 / 10", 76, 470, { size: 14, color: C.muted, alpha: a * seg(k, 1.6, 2.1) });
}

function step2(ctx, k, a) {
  text(ctx, "Six components, trained apart", 76, 230, { size: 22, weight: 650, alpha: a });
  text(ctx, "Backbone size and one-run training time. The gap is large.", 76, 258, { size: 15, color: C.soft, alpha: a });
  const mins = [134, 96, 15, 3, 0.5, 0.08];
  D.cost.forEach(([nm, par, tm], i) => {
    const s = easeOut(seg(k, 0.2 + i * 0.2, 0.8 + i * 0.2));
    const y = 305 + i * 46;
    text(ctx, nm, 76, y + 6, { size: 16, alpha: a * s });
    text(ctx, par === "-" ? "no backbone" : par + " params", 250, y + 6, { size: 14, color: C.muted, alpha: a * s });
    rrect(ctx, 400, y - 9, Math.max(6, 520 * (mins[i] / 134) * s), 18, 9, i < 2 ? C.accent : C.ramp[2], null, a);
    text(ctx, tm, 400 + Math.max(6, 520 * (mins[i] / 134) * s) + 12, y + 6, { size: 15, weight: 700, alpha: a * s });
  });
}

function step3(ctx, k, a) {
  text(ctx, "One score matrix per component", 76, 230, { size: 22, weight: 650, alpha: a });
  text(ctx, `${Z.val} validation notes by ${Z.codes} codes. The search then works on these cached matrices, not on text.`, 76, 258, { size: 15, color: C.soft, alpha: a });
  const names = D.components.map((c) => c[0]);
  names.forEach((nm, i) => {
    const s = easeOut(seg(k, 0.2 + i * 0.15, 0.8 + i * 0.15));
    const x = 76 + i * 176;
    const y = 320 + (1 - s) * 16;
    rrect(ctx, x, y, 150, 170, 8, "#ffffff", C.rule, a * s);
    ctx.save();
    ctx.globalAlpha = a * s;
    ctx.strokeStyle = C.gridStrong;
    ctx.beginPath();
    for (let g = 1; g < 10; g++) {
      ctx.moveTo(x + g * 15, y);
      ctx.lineTo(x + g * 15, y + 170);
      ctx.moveTo(x, y + g * 17);
      ctx.lineTo(x + 150, y + g * 17);
    }
    ctx.stroke();
    ctx.restore();
    text(ctx, nm, x + 75, y + 196, { size: 14, weight: 600, align: "center", alpha: a * s });
  });
  text(ctx, `each grid: ${Z.val} × ${Z.codes} (shape only, not values)`, 76, 552, { size: 13, color: C.muted, alpha: a * seg(k, 1.4, 1.9) });
}

function step4(ctx, k, a) {
  text(ctx, "Search over fusion strategies", 76, 230, { size: 22, weight: 650, alpha: a });
  text(ctx, "11 fusion strategies, 3 composition operators. Random restarts, hill-climbing and Variable Neighbourhood Search.", 76, 258, { size: 15, color: C.soft, alpha: a });
  const [cw, ch, gx, gy] = [54, 40, 76, 310];
  const sweep = seg(k, 0.3, 2.0);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 11; c++) {
      const on = (c + r * 0.5) / 12 < sweep;
      const best = r === 0 && c === 4;
      const win = best && k > 2.2;
      rrect(ctx, gx + c * (cw + 8), gy + r * (ch + 8), cw, ch, 8, win ? C.accent : on ? C.ramp[1 + ((c + r * 3) % 3)] : "#ffffff", win ? null : C.rule, a);
    }
  }
  text(ctx, "illustrative grid: strategies across, operators down; one cell wins", 76, 470, { size: 13, color: C.muted, alpha: a * seg(k, 1.0, 1.5) });
  const big = easeOut(seg(k, 0.8, 1.4));
  text(ctx, "10,000 evaluations × 2 restarts × 2 optimisers", 76, 500, { size: 20, weight: 650, alpha: a * big });
  text(ctx, "under 5 minutes on CPU", 76, 530, { size: 16, color: C.accentDeep, weight: 650, alpha: a * big });
}

function step5(ctx, k, a) {
  text(ctx, "ICD-10 codes for each note", 76, 230, { size: 22, weight: 650, alpha: a });
  text(ctx, "The final system is the best fusion found. It ranked first in the shared task.", 76, 258, { size: 15, color: C.soft, alpha: a });
  const s = easeOut(seg(k, 0.3, 0.9));
  rrect(ctx, 76, 310, 420, 170, 16, "#ffffff", C.accent, a * s);
  text(ctx, "official test set micro-F1", 104, 352, { size: 15, color: C.soft, alpha: a * s });
  text(ctx, D.test[0][3].toFixed(4), 104, 428, { size: 64, weight: 700, color: C.accent, alpha: a * s });
  const tag = easeOut(seg(k, 1.1, 1.7));
  rrect(ctx, 540, 310, 300, 80, 16, C.accentSoft, null, a * tag);
  text(ctx, "1st place", 690, 360, { size: 30, weight: 700, color: C.accentDeep, align: "center", alpha: a * tag });
  text(ctx, "BioASQ / CLEF ELCardioCC 2026", 540, 424, { size: 15, color: C.soft, alpha: a * tag });
  text(ctx, "Greek cardiology discharge summaries", 540, 448, { size: 15, color: C.soft, alpha: a * tag });
}

start((ctx, t) => {
  head(ctx, "How CardioICD works", "From free-text notes to ICD-10 codes in five steps.");
  chips(ctx, t);
  [step1, step2, step3, step4, step5].forEach((f, i) => {
    const a = seg(t, i * SEC, i * SEC + FADE) * (i < 4 ? 1 - seg(t, (i + 1) * SEC - FADE, (i + 1) * SEC) : 1);
    if (a > 0) f(ctx, t - i * SEC, a);
  });
});
