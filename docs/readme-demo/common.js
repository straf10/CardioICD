// Shared helpers for the README demo scenes. A scene draws frame t (seconds) with seek(t).
// render.py calls seek for every frame. Open a scene with ?play to watch it live.
const C = {
  bg: "#ffffff",
  panel: "#f5f8fb",
  grid: "#e8eef4",
  gridStrong: "#d5dfe9",
  rule: "#c5d2de",
  ink: "#0b2540",
  soft: "#3b5470",
  muted: "#6e8499",
  accent: "#1a6fb5",
  accentDeep: "#0b4f8a",
  accentSoft: "#dce9f5",
  slate: "#8aa0b5",
  // sequential accent ramp, light to deep
  ramp: ["#e6eff8", "#bcd4ea", "#85b0d8", "#3f84c0", "#0b4f8a"],
};
const W = 1200;
const H = 675;
const FONT = '"Segoe UI", "Inter", system-ui, sans-serif';
const MONO = '"Cascadia Mono", Consolas, monospace';

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const lerp = (a, b, k) => a + (b - a) * k;
const seg = (t, a, b) => clamp((t - a) / (b - a));
const easeOut = (k) => 1 - Math.pow(1 - k, 3);
const easeInOut = (k) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);

function rng(seed) {
  let a = seed >>> 0; // mulberry32: same frame on every render
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function normal(r) {
  return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
}

function text(ctx, s, x, y, o = {}) {
  ctx.save();
  ctx.globalAlpha = o.alpha ?? 1;
  ctx.fillStyle = o.color ?? C.ink;
  ctx.font = `${o.weight ?? 500} ${o.size ?? 16}px ${o.mono ? MONO : FONT}`;
  ctx.textAlign = o.align ?? "left";
  ctx.textBaseline = o.base ?? "alphabetic";
  ctx.letterSpacing = `${o.spacing ?? 0}px`;
  ctx.fillText(s, x, y);
  ctx.restore();
}

function rrect(ctx, x, y, w, h, r, fill, stroke, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.rect(x, y, w, h); // square corners: r is kept so callers stay unchanged
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }
  ctx.restore();
}

// Square marker, used wherever a dot would be.
function sq(ctx, x, y, r) {
  ctx.rect(x - r, y - r, 2 * r, 2 * r);
}

// Title block: small accent tick, title, one plain caption line.
function head(ctx, title, sub, alpha = 1, step = "") {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = C.accent;
  ctx.fillRect(56, 44, 5, 56);
  ctx.restore();
  if (step) text(ctx, step, 76, 54, { size: 12, weight: 700, color: C.accent, alpha, spacing: 2 });
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = C.rule;
  ctx.fillRect(56, 120, W - 112, 1);
  ctx.restore();
  text(ctx, title, 76, 82, { size: 30, weight: 650, alpha });
  text(ctx, sub, 76, 106, { size: 15, color: C.soft, alpha });
}

function start(draw, w = W, h = H) {
  const canvas = document.getElementById("c");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const seek = (t) => {
    ctx.save();
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, h);
    draw(ctx, t);
    ctx.restore();
  };
  window.seek = seek;
  document.fonts.ready.then(() => {
    seek(0);
    window.ready = true;
    if (location.search.includes("play")) {
      const t0 = performance.now();
      const loop = () => {
        seek(((performance.now() - t0) / 1000) % (window.DURATION || 12));
        requestAnimationFrame(loop);
      };
      loop();
    }
  });
}

function wrapWords(ctx, s, maxW, size, weight = 400) {
  ctx.font = `${weight} ${size}px ${FONT}`;
  const lines = [];
  let cur = [];
  let curW = 0;
  const sp = ctx.measureText(" ").width;
  let idx = 0;
  for (const word of s.split(" ")) {
    const w = ctx.measureText(word).width;
    if (cur.length && curW + sp + w > maxW) {
      lines.push(cur);
      cur = [];
      curW = 0;
    }
    cur.push({ word, start: idx, w, x: curW ? curW + sp : 0 });
    curW += (curW ? sp : 0) + w;
    idx += word.length + 1;
  }
  lines.push(cur);
  return lines;
}
