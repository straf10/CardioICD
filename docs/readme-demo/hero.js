// Hero: tune the threshold, fuse six components, read one synthetic note.
window.DURATION = 17;
const D = window.DATA;
const B1 = [0, 5.8];
const B2 = [5.8, 10];
const B3 = [10, 17];
const FADE = 0.4;
const vis = (t, [a, b]) => seg(t, a, a + FADE) * (1 - seg(t, b - FADE, b));

// ================= beat 1: threshold tuning =================
function beat1(ctx, t, a) {
  const k = t - B1[0];
  head(ctx, "Tune the threshold", "One cut-off for all codes is a blunt tool. Each code gets its own.", a, "1 · TUNE");
  const T = D.thr;
  const [x0, x1, y0, y1] = [100, 760, 175, 425];
  const [lo, hi] = [0.7, 0.82];
  const X = (v) => lerp(x0, x1, (v - 0.45) / 0.3);
  const Y = (v) => lerp(y1, y0, (v - lo) / (hi - lo));
  ctx.save();
  ctx.globalAlpha = a;
  rrect(ctx, x0 - 44, y0 - 50, x1 - x0 + 70, y1 - y0 + 215, 14, "rgba(255,255,255,0.92)", C.rule);
  for (const v of [0.7, 0.74, 0.78, 0.82]) {
    ctx.strokeStyle = C.gridStrong;
    ctx.beginPath();
    ctx.moveTo(x0, Y(v));
    ctx.lineTo(x1, Y(v));
    ctx.stroke();
    text(ctx, v.toFixed(2), x0 - 10, Y(v) + 4, { size: 12, color: C.muted, align: "right" });
  }
  for (const v of [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]) text(ctx, v.toFixed(2), X(v), y1 + 20, { size: 12, color: C.muted, align: "center" });
  text(ctx, "global threshold t", (x0 + x1) / 2, y1 + 40, { size: 13, color: C.soft, align: "center" });
  text(ctx, "validation micro-F1", x0 - 30, y0 - 26, { size: 13, color: C.soft });

  // curve
  const p = easeInOut(seg(k, 0.3, 2.6));
  const n = Math.round(p * (T.x.length - 1));
  ctx.strokeStyle = C.accent;
  ctx.lineWidth = 3;
  ctx.lineJoin = "miter";
  ctx.beginPath();
  for (let i = 0; i <= n; i++) (i ? ctx.lineTo(X(T.x[i]), Y(T.f1[i])) : ctx.moveTo(X(T.x[i]), Y(T.f1[i])));
  ctx.stroke();
  const mark = (xv, yv, col, lab, dy, al, al2 = "center", dx = 0) => {
    ctx.globalAlpha = a * al;
    ctx.fillStyle = col;
    ctx.beginPath();
    sq(ctx, X(xv), Y(yv), 6);
    ctx.fill();
    text(ctx, lab, X(xv) + dx, Y(yv) + dy, { size: 14, weight: 650, color: col, align: al2, alpha: a * al });
  };
  mark(0.5, T.at05, C.slate, `t = 0.50  ·  ${T.at05.toFixed(4)}`, 28, easeOut(seg(k, 1.2, 1.6)) * (n >= 5 ? 1 : 0), "left", 14);
  mark(T.best_x, T.best_f1, C.accentDeep, `t = ${T.best_x}  ·  ${T.best_f1.toFixed(4)}`, -16, easeOut(seg(k, 2.6, 3.0)), "center");

  // 115 codes, one dot each. Most stay at the best global value; the rest move to their own.
  const sy = 530;
  ctx.globalAlpha = a;
  ctx.strokeStyle = C.rule;
  ctx.beginPath();
  ctx.moveTo(x0, sy + 22);
  ctx.lineTo(x1, sy + 22);
  ctx.stroke();
  const moved = T.per_label.filter((v) => Math.abs(v - T.best_x) > 1e-9);
  const mv = easeInOut(seg(k, 3.4, 4.7));
  let m = 0;
  T.per_label.forEach((v, i) => {
    const cx0 = X(T.best_x) + ((i % 13) - 6) * 5;
    const cy0 = sy - Math.floor(i / 13) * 5 + 8;
    const isMoved = Math.abs(v - T.best_x) > 1e-9;
    const jitter = isMoved ? (m++ % 4) * 7 - 6 : 0;
    const px = isMoved ? lerp(cx0, X(v), mv) : cx0;
    const py = isMoved ? lerp(cy0, sy + 8 + jitter, mv) : cy0;
    ctx.fillStyle = isMoved ? C.accent : C.slate;
    ctx.globalAlpha = a * easeOut(seg(k, 3.0 + (i % 23) * 0.012, 3.35));
    ctx.beginPath();
    sq(ctx, px, py, 1.8);
    ctx.fill();
  });
  text(ctx, `${T.labels.length} codes, one dot each`, x0, sy + 44, { size: 13, color: C.soft, alpha: a * seg(k, 3.2, 3.6) });
  text(ctx, `${moved.length} codes leave ${T.best_x}`, X(0.75), sy + 44, { size: 13, weight: 650, color: C.accent, align: "right", alpha: a * seg(k, 4.5, 4.9) });
  ctx.restore();

  // result tiles
  const tiles = [
    ["fixed t = 0.50", T.at05, C.ink, 1.0],
    [`best single t = ${T.best_x}`, T.best_f1, C.accentDeep, 2.7],
    ["a threshold per code", T.tuned, C.accent, 4.6],
  ];
  tiles.forEach(([lab, v, col, at], i) => {
    const s = easeOut(seg(k, at, at + 0.5));
    const y = 178 + i * 112 + (1 - s) * 14;
    rrect(ctx, 820, y, 324, 98, 14, "#ffffff", i === 2 ? C.accent : C.rule, a * s);
    text(ctx, lab, 842, y + 32, { size: 14, color: C.soft, alpha: a * s });
    text(ctx, v.toFixed(4), 842, y + 76, { size: 38, weight: 700, color: col, alpha: a * s });
  });
  text(ctx, "Greek BERT, validation split (250 notes), local rerun", 820, 540, { size: 13, color: C.muted, alpha: a * seg(k, 4.8, 5.2) });
}

// ================= beat 2: fuse =================
function beat2(ctx, t, a) {
  const k = t - B2[0];
  head(ctx, "Fuse six components", "Each model errs in a different place. A search finds how to combine them.", a, "2 · FUSE");
  const [x0, x1] = [260, 520];
  const X = (v) => lerp(x0, x1, (v - 0.6) / 0.25);
  text(ctx, "validation micro-F1, one model each", 76, 160, { size: 13, weight: 650, color: C.soft, alpha: a });
  D.components.forEach(([nm, v], i) => {
    const s = easeOut(seg(k, 0.2 + i * 0.22, 0.7 + i * 0.22));
    const y = 205 + i * 50;
    text(ctx, nm, 76, y + 5, { size: 16, alpha: a * s });
    ctx.save();
    ctx.globalAlpha = a * s;
    ctx.strokeStyle = C.accentSoft;
    ctx.lineWidth = 8;
    ctx.lineCap = "butt";
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(X(v), y);
    ctx.stroke();
    ctx.fillStyle = C.accent;
    ctx.beginPath();
    sq(ctx, X(v), y, 7);
    ctx.fill();
    ctx.restore();
    text(ctx, v.toFixed(4), x1 + 36, y + 5, { size: 14, weight: 600, color: C.soft, alpha: a * s });
  });
  // fusion node
  const fa = easeOut(seg(k, 1.9, 2.4));
  ctx.save();
  ctx.globalAlpha = a * fa;
  ctx.strokeStyle = C.accent;
  ctx.fillStyle = C.accent;
  ctx.lineWidth = 2.5;
  for (const [xa, xb] of [[606, 624], [784, 820]]) {
    ctx.beginPath();
    ctx.moveTo(xa, 350);
    ctx.lineTo(xb - 4, 350);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(xb, 350);
    ctx.lineTo(xb - 9, 344);
    ctx.lineTo(xb - 9, 356);
    ctx.fill();
  }
  ctx.restore();
  rrect(ctx, 628, 305, 150, 90, 16, C.accent, null, a * fa);
  text(ctx, "ensemble", 703, 342, { size: 17, weight: 700, color: "#fff", align: "center", alpha: a * fa });
  text(ctx, "search", 703, 366, { size: 15, color: "#fff", align: "center", alpha: a * fa });
  // official test
  const [tx0, tx1] = [880, 1120];
  const TX = (v) => lerp(tx0, tx1, (v - 0.82) / 0.06);
  const rb = easeOut(seg(k, 2.5, 3.1));
  text(ctx, "official test set, micro-F1", 830, 160, { size: 13, weight: 650, color: C.soft, alpha: a * rb });
  [["Greek BERT", D.test[3][3], C.slate, 232], ["Ensemble", D.test[0][3], C.accent, 312]].forEach(([nm, v, col, y], i) => {
    const s = easeOut(seg(k, 2.5 + i * 0.6, 3.1 + i * 0.6));
    text(ctx, nm, 830, y - 12, { size: 16, alpha: a * s });
    ctx.save();
    ctx.globalAlpha = a * s;
    ctx.strokeStyle = C.accentSoft;
    ctx.lineWidth = 8;
    ctx.lineCap = "butt";
    ctx.beginPath();
    ctx.moveTo(tx0 - 40, y + 14);
    ctx.lineTo(TX(v), y + 14);
    ctx.stroke();
    ctx.fillStyle = col;
    ctx.beginPath();
    sq(ctx, TX(v), y + 14, 8);
    ctx.fill();
    ctx.restore();
    text(ctx, v.toFixed(4), TX(v), y + 44, { size: 16, weight: 700, color: col === C.slate ? C.ink : C.accentDeep, align: "center", alpha: a * s });
  });
  const gap = easeOut(seg(k, 3.3, 3.8));
  rrect(ctx, 880, 400, 190, 48, 24, C.accentSoft, null, a * gap);
  text(ctx, "+1.8 points", 975, 432, { size: 20, weight: 700, color: C.accentDeep, align: "center", alpha: a * gap });
  text(ctx, "Greek BERT: 100% of the data. Ensemble: merge_and.", 830, 492, { size: 12, color: C.muted, alpha: a * gap });
}

// ================= beat 3: read a note =================
const N = D.note;
const SHORT = { "Retrieval (IR)": "Retrieval", "NER + entity linking": "NER + EL" };
function beat3(ctx, t, a) {
  const k = t - B3[0];
  head(ctx, "Read the note, propose the codes", "Each highlighted span resolves into an ICD-10 code.", a, "3 · READ");
  const [px, py, pw] = [56, 140, 650];
  rrect(ctx, px, py, pw, 410, 14, "#ffffff", C.rule, a);
  const lines = wrapWords(ctx, N.text, pw - 44, 15);
  const LH = 24;
  const reveal = N.spans.map((_, i) => 0.5 + i * 0.42);
  lines.forEach((line, li) => {
    const y = py + 40 + li * LH;
    line.forEach((w) => {
      ctx.font = `400 15px ${FONT}`;
      for (let s = 0; s < N.spans.length; s++) {
        const [sa, sb] = N.spans[s];
        const from = Math.max(sa, w.start) - w.start;
        const to = Math.min(sb, w.start + w.word.length) - w.start;
        if (to <= from) continue;
        const pre = ctx.measureText(w.word.slice(0, from)).width;
        const wid = ctx.measureText(w.word.slice(from, to)).width;
        const al = easeOut(seg(k, reveal[s], reveal[s] + 0.3));
        ctx.save();
        ctx.globalAlpha = a * al * 0.9;
        ctx.fillStyle = C.accentSoft;
        ctx.fillRect(px + 22 + w.x + pre - 2, y - 16, wid + 4, 22);
        ctx.fillStyle = C.accent;
        ctx.fillRect(px + 22 + w.x + pre - 2, y + 4, (wid + 4) * al, 2);
        ctx.restore();
      }
      text(ctx, w.word, px + 22 + w.x, y, { size: 15, weight: 400, alpha: a, color: C.ink });
    });
  });
  // code list, in order of first evidence
  const order = [];
  N.spans.forEach(([, , c], i) => {
    if (!order.find((o) => o[0] === c)) order.push([c, reveal[i]]);
  });
  const rest = N.rows.filter(([c]) => !order.find((o) => o[0] === c));
  rest.forEach(([c], i) => order.push([c, 5.6 + i * 0.35]));
  const by = Object.fromEntries(N.rows.map(([c, m]) => [c, SHORT[m] || m]));
  text(ctx, "ICD-10 codes", 740, 160, { size: 13, weight: 650, color: C.soft, alpha: a });
  order.forEach(([c, at], i) => {
    const s = easeOut(seg(k, at + 0.15, at + 0.6));
    const y = 176 + i * 42 + (1 - s) * 10;
    rrect(ctx, 740, y, 404, 34, 10, "#ffffff", C.rule, a * s);
    text(ctx, c, 756, y + 23, { size: 15, weight: 700, color: C.accentDeep, mono: true, alpha: a * s });
    text(ctx, by[c], 1128, y + 23, { size: 14, color: C.soft, align: "right", alpha: a * s });
  });
  const done = easeOut(seg(k, 6.0, 6.5));
  text(ctx, `${order.length} codes · micro-F1 ${N.f1.toFixed(3)} on this note`, 740, 590, { size: 17, weight: 650, color: C.accentDeep, alpha: a * done });
  text(ctx, "Synthetic note for illustration. No real patient text.", 56, 590, { size: 13, color: C.muted, alpha: a });
}

start((ctx, t) => {
  const a1 = vis(t, B1);
  const a2 = vis(t, B2);
  const a3 = vis(t, B3);
  if (a1 > 0) beat1(ctx, t, a1);
  if (a2 > 0) beat2(ctx, t, a2);
  if (a3 > 0) beat3(ctx, t, a3);
  [B1, B2, B3].forEach((b, i) => {
    ctx.fillStyle = t >= b[0] && t < b[1] ? C.accent : C.rule;
    ctx.beginPath();
    sq(ctx, 1100 + i * 18, 52, 5);
    ctx.fill();
  });
});
