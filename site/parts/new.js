/* ---------------------------------------------------------------- page chrome */
const DS = DATA.describe;
const topbar = document.getElementById("topbar"), heroEl = document.querySelector(".hero");
function chrome() { topbar.classList.toggle("solid", scrollY > heroEl.offsetHeight - 70); }
addEventListener("scroll", chrome, { passive: true }); chrome();

/* blocks below the first screen fade in once; everything above it is visible at rest */
if (!reduce) {
  const pend = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.remove("pending"); pend.unobserve(e.target); } }), { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach(n => { if (n.getBoundingClientRect().top > innerHeight) { n.classList.add("pending"); pend.observe(n); } });
}

function rgba(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; }
function countTo(node, target, fmt, dur) {
  if (reduce) { node.textContent = fmt(target); return; }
  const t0 = performance.now();
  (function f(now) { const t = Math.min(1, (now - t0) / (dur || 1500)); node.textContent = fmt(target * ease(t)); if (t < 1) requestAnimationFrame(f); })(t0);
}

/* ---------------------------------------------------------------- hero */
const AG = TH.day14.groups.age_band;
countTo(document.getElementById("sOlder"), AG["35+"].missed_at_risk, v => Math.round(v * 100) + "%");
countTo(document.getElementById("sYounger"), AG["0-35"].missed_at_risk, v => Math.round(v * 100) + "%");
countTo(document.getElementById("sN"), S.day14.n_test, v => nf(Math.round(v)), 1800);

(function heroField() {
  const cv = document.getElementById("heroCanvas"), ctx = cv.getContext("2d");
  let W = 0, H = 0, dots = [];
  let seed = 3; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  function size() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(Math.min(260, W * H / 5200));
    seed = 3; dots = [];
    for (let i = 0; i < n; i++) dots.push({ x: rnd() * W, y: rnd() * H, r: 1.4 + rnd() * 2.6, v: 0.12 + rnd() * 0.35, risk: rnd() < 0.41, missed: rnd() < 0.3, ph: rnd() * 6.28 });
  }
  function frame(now) {
    const s1 = css("--s1"), s2 = css("--s2");
    ctx.clearRect(0, 0, W, H);
    const g = ctx.createRadialGradient(W * 0.78, H * 0.35, 10, W * 0.78, H * 0.35, Math.max(W, H) * 0.6);
    g.addColorStop(0, rgba(s1, 0.16)); g.addColorStop(1, rgba(s1, 0));
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    for (const d of dots) {
      if (!reduce) { d.y -= d.v; d.x += Math.sin(now / 2400 + d.ph) * 0.15; if (d.y < -8) { d.y = H + 8; d.x = Math.random() * W; } }
      const fade = Math.min(1, Math.max(0, (H - d.y) / 120)) * Math.min(1, d.y / 140);
      ctx.globalAlpha = (d.x > W * 0.45 ? 0.85 : 0.35) * fade;
      if (d.risk && d.missed) {
        ctx.strokeStyle = s2; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(d.x, d.y, d.r + 1.6, 0, 6.29); ctx.stroke();
      } else {
        ctx.fillStyle = d.risk ? s2 : s1; ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, 6.29); ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    if (!reduce) requestAnimationFrame(frame);
  }
  size(); addEventListener("resize", () => { size(); if (reduce) frame(0); });
  requestAnimationFrame(frame);
})();

/* ---------------------------------------------------------------- scrollytelling unit chart */
const scrolly = (function () {
  const U = DS.units, N = U.p.length;
  const cv = document.getElementById("unitCanvas"), ctx = cv.getContext("2d");
  const legend = document.getElementById("glegend");
  const chosen = TH.day14.threshold;
  let W = 0, H = 0, step = -1, anim = null;
  const cur = { x: new Float32Array(N), y: new Float32Array(N), r: 3, t: 0.5, tA: 0, rows: 0 };
  let from = null, to = null;
  let rs = 5; const rnd = () => (rs = (rs * 48271) % 2147483647) / 2147483647;
  const shuffled = Array.from({ length: N }, (_, i) => i).map(i => [rnd(), i]).sort((a, b) => a[0] - b[0]).map(a => a[1]);
  const delay = new Float32Array(N); for (let i = 0; i < N; i++) delay[i] = rnd() * 0.35;
  const missRate = t => CV.missed.all[Math.round(t * 100)];

  function texts() {
    const n = S.day14.n_test, pr = S.day14.pass_rate_test;
    document.getElementById("st0").textContent = nf(n) + " students";
    document.getElementById("st1").textContent = pct(pr) + " passed · " + pct(1 - pr) + " didn't";
    document.getElementById("st2").textContent = "AUROC " + f3(S.day14.later_run.boosted.auc);
    document.getElementById("st3").textContent = pct(CV.missed.all[50]) + " of at-risk missed";
    document.getElementById("st4").textContent = pct(TH.day14.missed_at_risk) + " missed · " + pct(TH.day14.flag_rate) + " flagged";
    document.getElementById("st5").textContent = pct(AG["35+"].missed_at_risk) + " vs " + pct(AG["0-35"].missed_at_risk) + " missed";
  }
  texts();

  function size() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function gridLayout(order) {
    const top = 50, bot = 34, side = 22, aw = W - 2 * side, ah = H - top - bot;
    const s = Math.sqrt(aw * ah / N);
    const cols = Math.floor(aw / s), rows = Math.ceil(N / cols);
    const sp = Math.min(aw / cols, ah / rows);
    const ox = side + (aw - cols * sp) / 2 + sp / 2, oy = top + (ah - rows * sp) / 2 + sp / 2;
    const x = new Float32Array(N), y = new Float32Array(N);
    order.forEach((i, k) => { x[i] = ox + (k % cols) * sp; y[i] = oy + Math.floor(k / cols) * sp; });
    return { x, y, r: sp * 0.36 };
  }
  function histLayout(byAge) {
    const top = 54, bot = 80, side = 30, aw = W - 2 * side;
    const bins = 40, cw = aw / bins;
    const rows = byAge ? [0, 1] : [null];
    const rowH = (H - top - bot) / rows.length;
    const stacks = rows.map(() => Array.from({ length: bins }, () => []));
    for (let i = 0; i < N; i++) {
      const ri = byAge ? U.older[i] : 0;
      const b = Math.min(bins - 1, Math.floor(U.p[i] * bins));
      stacks[ri][b].push(i);
    }
    let maxS = 1;
    stacks.forEach(r => r.forEach(c => { c.sort((a, b) => (U.passed[b] - U.passed[a]) || (U.p[a] - U.p[b])); maxS = Math.max(maxS, c.length); }));
    const per = Math.max(1, Math.round(Math.sqrt(maxS * cw / (rowH * 0.92))));
    const sx = cw / per, sy = Math.min(sx, (rowH * 0.9) / Math.ceil(maxS / per));
    const x = new Float32Array(N), y = new Float32Array(N);
    stacks.forEach((r, ri) => {
      const base = top + rowH * (ri + 1) - 6;
      r.forEach((c, b) => c.forEach((i, k) => {
        x[i] = side + b * cw + (k % per) * sx + sx / 2;
        y[i] = base - Math.floor(k / per) * sy - sy / 2;
      }));
    });
    return { x, y, r: Math.max(1.2, Math.min(sx, sy) * 0.4), rows: rows.length, side, aw, top, rowH };
  }
  let lay = null;
  function layoutFor(s) {
    if (s <= 0) return gridLayout(shuffled);
    if (s === 1) return gridLayout(Array.from({ length: N }, (_, i) => i).sort((a, b) => (U.passed[b] - U.passed[a]) || (shuffled.indexOf(a) - shuffled.indexOf(b))));
    return histLayout(s >= 5);
  }
  function thresholdFor(s) { return s >= 4 ? chosen : 0.5; }

  function legendFor(s) {
    legend.replaceChildren();
    const add = (color, label, ring) => {
      const sp = document.createElement("span");
      const k = document.createElement("span"); k.className = "dotkey";
      if (ring) { k.style.border = "2px solid " + color; k.style.background = "transparent"; } else k.style.background = color;
      sp.appendChild(k); sp.appendChild(document.createTextNode(label)); legend.appendChild(sp);
    };
    if (s === 0) return add(css("--muted"), "A student");
    add(css("--s1"), "Passed"); add(css("--s2"), "Failed or withdrew");
    if (s >= 3) add(css("--s2"), "At risk but not flagged", true);
  }

  function draw() {
    if (!W) return;
    const s1 = css("--s1"), s2 = css("--s2"), muted = css("--muted"), ink = css("--ink"), ink2 = css("--ink2"), grid = css("--grid"), surface = css("--surface");
    ctx.clearRect(0, 0, W, H);
    const L = lay;
    const showT = step >= 3 && L && L.aw;
    if (L && L.aw) {
      // axis per row
      for (let r = 0; r < L.rows; r++) {
        const base = L.top + L.rowH * (r + 1) - 6 + 4;
        ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L.side, base); ctx.lineTo(L.side + L.aw, base); ctx.stroke();
        if (L.rows > 1) {
          ctx.fillStyle = ink; ctx.font = "600 13px Inter, system-ui"; ctx.textAlign = "left";
          const g = r === 0 ? AG["0-35"] : AG["35+"];
          ctx.fillText((r === 0 ? "Under 35" : "35 and over") + " · " + pct(g.missed_at_risk) + " of at-risk missed", L.side, L.top + L.rowH * r + 14);
        }
      }
      ctx.fillStyle = muted; ctx.font = "12px 'IBM Plex Mono', monospace"; ctx.textAlign = "center";
      // tick labels sit just under the axis; the caption shares the bottom line with the dot note (right)
      const by = H - 58;
      [0, 0.25, 0.5, 0.75, 1].forEach((v, k) => { ctx.textAlign = k === 0 ? "left" : k === 4 ? "right" : "center"; ctx.fillText(v.toFixed(2), L.side + v * L.aw, by); });
      ctx.textAlign = "left";
      if (W > 560) ctx.fillText("predicted chance of passing →", L.side, H - 16);
    }
    if (showT) {
      const tx = L.side + cur.t * L.aw;
      ctx.globalAlpha = cur.tA;
      ctx.fillStyle = rgba(s2, 0.08); ctx.fillRect(L.side, 44, tx - L.side, H - 80 - 44);
      ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(tx, 40); ctx.lineTo(tx, H - 76); ctx.stroke();
      ctx.fillStyle = ink; ctx.beginPath(); ctx.roundRect ? ctx.roundRect(tx - 24, 30, 48, 20, 10) : ctx.rect(tx - 24, 30, 48, 20); ctx.fill();
      ctx.fillStyle = surface; ctx.font = "600 12px Inter, system-ui"; ctx.textAlign = "center"; ctx.fillText(cur.t.toFixed(2), tx, 44);
      ctx.textAlign = "left"; ctx.fillStyle = ink2; ctx.font = "12px Inter, system-ui";
      if (tx - L.side > 70 && L.rows === 1) ctx.fillText("flagged", L.side + 6, 64);
      ctx.globalAlpha = 1;
    }
    const r = cur.r;
    for (let i = 0; i < N; i++) {
      const risk = !U.passed[i];
      let col = step <= 0 ? muted : (risk ? s2 : s1), a = 1, ring = false;
      if (showT) {
        const flagged = U.p[i] < cur.t;
        if (risk && !flagged) ring = true;
        else if (risk) a = 0.55;
        else a = flagged ? 0.5 : 0.22;
      }
      ctx.globalAlpha = a;
      if (ring) { ctx.strokeStyle = s2; ctx.lineWidth = Math.max(1, r * 0.55); ctx.beginPath(); ctx.arc(cur.x[i], cur.y[i], r * 0.85, 0, 6.29); ctx.stroke(); }
      else { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(cur.x[i], cur.y[i], r, 0, 6.29); ctx.fill(); }
    }
    ctx.globalAlpha = 1;
  }

  function go(s) {
    if (s === step && lay) return;
    step = s; legendFor(s);
    lay = layoutFor(s);
    const target = { x: lay.x, y: lay.y, r: lay.r, t: thresholdFor(s), tA: s >= 3 ? 1 : 0 };
    if (reduce || !from) {
      cur.x.set(target.x); cur.y.set(target.y); cur.r = target.r; cur.t = target.t; cur.tA = target.tA;
      from = true; draw(); return;
    }
    const fx = cur.x.slice(), fy = cur.y.slice(), fr = cur.r, ft = cur.t, fa = cur.tA;
    const t0 = performance.now(), dur = 1300;
    if (anim) cancelAnimationFrame(anim);
    (function f(now) {
      const g = (now - t0) / dur;
      for (let i = 0; i < N; i++) {
        const k = Math.min(1, Math.max(0, (g - delay[i]) / 0.65)), e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        cur.x[i] = lerp(fx[i], target.x[i], e); cur.y[i] = lerp(fy[i], target.y[i], e);
      }
      const e = ease(Math.min(1, g));
      cur.r = lerp(fr, target.r, e); cur.t = lerp(ft, target.t, e); cur.tA = lerp(fa, target.tA, e);
      draw();
      if (g < 1) anim = requestAnimationFrame(f); else anim = null;
    })(t0);
  }

  const steps = [...document.querySelectorAll(".step")];
  const so = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    steps.forEach(x => x.classList.toggle("on", x === e.target));
    go(+e.target.dataset.step);
  }), { rootMargin: "-45% 0px -45% 0px" });
  steps.forEach(x => so.observe(x));
  size(); go(0);
  // deep link: #step-3 opens the story at that step
  const m = /^#step-([0-5])$/.exec(location.hash);
  if (m) { const k = +m[1]; steps.forEach(x => x.classList.toggle("on", +x.dataset.step === k)); go(k); requestAnimationFrame(() => steps[k].scrollIntoView({ block: "center" })); }
  addEventListener("resize", () => { size(); const s = step; step = -1; from = null; go(s); });
  return { redraw: () => { legendFor(step); draw(); } };
})();

/* ---------------------------------------------------------------- daily clicks */
const clickLines = (function () {
  const svg = document.getElementById("clickSvg"), C = DS.clicks, days = C.days;
  const panels = [["young", "Under 35"], ["older", "35 and over"]];
  const max = Math.max(...["pass_young", "risk_young", "pass_older", "risk_older"].flatMap(g => C[g]));
  let prog = 0;
  function draw(p) {
    clear(svg);
    panels.forEach(([g, title], pi) => {
      const X0 = 50 + pi * 375, X1 = X0 + 320, Y0 = 250, Y1 = 40;
      const x = d => X0 + (d - days[0]) / (days[days.length - 1] - days[0]) * (X1 - X0);
      const y = v => Y0 - v / max * (Y0 - Y1);
      txt(svg, X0, 22, title, "lbl-strong");
      el("rect", { x: x(0), y: Y1 - 6, width: x(13.5) - x(0), height: Y0 - Y1 + 6, fill: css("--wash1") }, svg);
      const gg = el("g", { class: "grid" }, svg);
      const step = max > 30 ? 10 : 5;
      for (let v = 0; v <= max; v += step) { el("line", { x1: X0, y1: y(v), x2: X1, y2: y(v) }, gg); if (pi === 0) txt(svg, X0 - 8, y(v) + 4, String(v), "lbl-muted", { "text-anchor": "end" }); }
      [-14, 0, 14, 27].forEach(d => txt(svg, x(d), Y0 + 18, d === 0 ? "day 0" : String(d), "lbl-muted", { "text-anchor": "middle" }));
      el("line", { x1: x(0), y1: Y1 - 6, x2: x(0), y2: Y0, stroke: css("--axis") }, svg);
      [["risk_" + g, css("--s2")], ["pass_" + g, css("--s1")]].forEach(([k, col]) => {
        const n = Math.max(2, Math.round(days.length * p));
        let d = ""; for (let i = 0; i < n; i++) d += (i ? "L" : "M") + x(days[i]) + " " + y(C[k][i]) + " ";
        let a = d + `L${x(days[n - 1])} ${Y0} L${x(days[0])} ${Y0} Z`;
        el("path", { d: a, fill: col, opacity: 0.08 }, svg);
        el("path", { d, fill: "none", stroke: col, "stroke-width": 2, "stroke-linejoin": "round" }, svg);
      });
      const hit = el("rect", { x: X0, y: Y1, width: X1 - X0, height: Y0 - Y1, class: "hit" }, svg);
      const cross = el("line", { x1: 0, y1: Y1, x2: 0, y2: Y0, stroke: css("--ink"), "stroke-width": 1, opacity: 0 }, svg);
      hit.addEventListener("pointermove", e => {
        const r = svg.getBoundingClientRect(), vx = (e.clientX - r.left) / r.width * 760;
        const i = Math.max(0, Math.min(days.length - 1, Math.round((vx - X0) / (X1 - X0) * (days.length - 1))));
        cross.setAttribute("x1", x(days[i])); cross.setAttribute("x2", x(days[i])); cross.setAttribute("opacity", 0.5);
        showTip(e.clientX, e.clientY, { title: `${title}, day ${days[i]}`, rows: [
          { value: C["pass_" + g][i].toFixed(1), label: `clicks, passed (n = ${nf(C.n["pass_" + g])})`, color: css("--s1") },
          { value: C["risk_" + g][i].toFixed(1), label: `clicks, failed or withdrew (n = ${nf(C.n["risk_" + g])})`, color: css("--s2") }] });
      });
      hit.addEventListener("pointerleave", () => { hideTip(); cross.setAttribute("opacity", 0); });
    });
  }
  draw(reduce ? 1 : 0.02);
  whenShown(svg, () => {
    if (reduce) return draw(1);
    const t0 = performance.now();
    (function f(now) { prog = ease(Math.min(1, (now - t0) / 1600)); draw(prog); if (prog < 1) requestAnimationFrame(f); })(t0);
  });
  return { redraw: () => draw(reduce ? 1 : Math.max(prog, 0.02)) };
})();

/* mean clicks a day in days 0..13 */
const clickBars = (function () {
  const svg = document.getElementById("clickBars"), C = DS.clicks;
  const i0 = C.days.indexOf(0), i1 = C.days.indexOf(13);
  const mean = k => C[k].slice(i0, i1 + 1).reduce((a, b) => a + b, 0) / (i1 - i0 + 1);
  const rows = [["Under 35", "young"], ["35 and over", "older"]];
  const vals = {}; ["pass_young", "risk_young", "pass_older", "risk_older"].forEach(k => vals[k] = mean(k));
  const max = Math.max(...Object.values(vals));
  const q = document.getElementById("clickQuote");
  q.replaceChildren();
  const add = (t, cls) => { const s = document.createElement("span"); if (cls) s.className = cls; s.textContent = t; q.appendChild(s); };
  add("In their first two weeks, at-risk students over 35 click "); add(vals.risk_older.toFixed(1) + " times a day", "o");
  add(", " + Math.round((vals.risk_older / vals.risk_young - 1) * 100) + "% more than at-risk students under 35 (" + vals.risk_young.toFixed(1) + "). A model that reads clicks takes that as a good sign, ");
  add("which may be why it misses more of them.", "b");
  let prog = 0;
  function draw(p) {
    clear(svg);
    const X0 = 140, X1 = 660, sc = v => (X1 - X0) * v / max;
    const g = el("g", { class: "grid" }, svg);
    [0, 5, 10, 15].filter(v => v <= max + 1).forEach(v => { el("line", { x1: X0 + sc(v), y1: 8, x2: X0 + sc(v), y2: 178 }, g); txt(svg, X0 + sc(v), 196, String(v), "lbl-muted", { "text-anchor": "middle" }); });
    rows.forEach(([name, g2], ri) => {
      const y0 = 18 + ri * 86;
      txt(svg, X0 - 14, y0 + 30, name, "lbl-strong", { "text-anchor": "end" });
      [["pass_" + g2, css("--s1"), "passed"], ["risk_" + g2, css("--s2"), "failed or withdrew"]].forEach(([k, col, lab], j) => {
        const y = y0 + j * 28, w = Math.max(0, sc(vals[k]) * p), h = 20;
        el("path", { d: `M${X0} ${y} h${Math.max(0, w - 4)} q4 0 4 4 v${h - 8} q0 4 -4 4 h${-Math.max(0, w - 4)} z`, fill: col }, svg);
        txt(svg, X0 + w + 8, y + 15, vals[k].toFixed(1), "lbl-strong", { opacity: p });
        const hit = el("rect", { x: 0, y: y - 3, width: 760, height: 26, class: "hit" }, svg);
        attachTip(hit, () => ({ title: `${name}, ${lab}`, rows: [{ value: vals[k].toFixed(2), label: "clicks a day, days 0–13", color: col }, { value: nf(C.n[k]), label: "students" }] }));
      });
    });
  }
  draw(reduce ? 1 : 0);
  whenShown(svg, () => {
    if (reduce) return draw(1);
    const t0 = performance.now();
    (function f(now) { prog = ease(Math.min(1, (now - t0) / 1100)); draw(prog); if (prog < 1) requestAnimationFrame(f); })(t0);
  });
  return { redraw: () => draw(reduce ? 1 : Math.max(prog, 1)) };
})();

/* ---------------------------------------------------------------- withdrawal timing */
const withdraw = (function () {
  const svg = document.getElementById("wdSvg"), W = DS.withdrawal_days, e = W.edges, c = W.counts;
  const max = Math.max(...c);
  const before = c.reduce((a, v, i) => a + (e[i + 1] <= 0 ? v : 0), 0);
  const X0 = 50, X1 = 730, Y0 = 200, Y1 = 30;
  const x = d => X0 + (d - e[0]) / (e[e.length - 1] - e[0]) * (X1 - X0);
  const y = v => Y0 - v / max * (Y0 - Y1);
  let prog = 0;
  function draw(p) {
    clear(svg);
    const g = el("g", { class: "grid" }, svg);
    [0, 500, 1000, 1500, 2000].filter(v => v <= max * 1.05).forEach(v => { el("line", { x1: X0, y1: y(v), x2: X1, y2: y(v) }, g); txt(svg, X0 - 8, y(v) + 4, nf(v), "lbl-muted", { "text-anchor": "end" }); });
    c.forEach((v, i) => {
      const bx = x(e[i]) + 1, bw = x(e[i + 1]) - x(e[i]) - 2, h = (Y0 - y(v)) * p;
      el("path", { d: `M${bx} ${Y0} v${-Math.max(0, h - 4)} q0 -4 4 -4 h${bw - 8} q4 0 4 4 v${Math.max(0, h - 4)} z`, fill: e[i + 1] <= 0 ? css("--s2") : rgba(css("--s2"), 0.55) }, svg);
      const hit = el("rect", { x: bx, y: Y1, width: bw, height: Y0 - Y1, class: "hit" }, svg);
      attachTip(hit, () => ({ title: `days ${e[i]} to ${e[i + 1] - 1}`, rows: [{ value: nf(v), label: "withdrawals", color: css("--s2") }] }));
    });
    el("line", { x1: X0, y1: Y0, x2: X1, y2: Y0, class: "base" }, svg);
    [-120, -60, 0, 60, 120, 180, 240].forEach(d => txt(svg, x(d), Y0 + 18, d === 0 ? "day 0" : String(d), "lbl-muted", { "text-anchor": "middle" }));
    [[0, "course starts"], [14, "14"], [28, "28"]].forEach(([d, lab], k) => {
      el("line", { x1: x(d), y1: Y1 - 10, x2: x(d), y2: Y0, stroke: css("--ink"), "stroke-width": k ? 1 : 1.5, opacity: k ? .45 : .8 }, svg);
      if (!k) txt(svg, x(d) + 6, Y1 - 12, lab, "lbl-muted");
    });
    txt(svg, X0, 242, `${pct(before / W.n)} of ${nf(W.n)} dated withdrawals happen before day 0. The tall bars from day 0 to 19 are the next wave.`, "lbl");
  }
  draw(reduce ? 1 : 0);
  whenShown(svg, () => {
    if (reduce) return draw(1);
    const t0 = performance.now();
    (function f(now) { prog = ease(Math.min(1, (now - t0) / 1100)); draw(prog); if (prog < 1) requestAnimationFrame(f); })(t0);
  });
  return { redraw: () => draw(reduce ? 1 : Math.max(prog, 1)) };
})();

/* ---------------------------------------------------------------- outcome mix */
const outcomes = (function () {
  const O = DS.outcomes.overall, total = Object.values(O).reduce((a, b) => a + b, 0);
  document.getElementById("outN").textContent = nf(total);
  const parts = [["Distinction", "#1c5cab", "#ffffff"], ["Pass", "#86b6ef", "#0b0b0b"], ["Fail", "#f3a07c", "#0b0b0b"], ["Withdrawn", "#c8501f", "#ffffff"]];
  const bar = document.getElementById("outBar"), leg = document.getElementById("outLegend");
  function build(grow) {
    bar.replaceChildren(); leg.replaceChildren();
    parts.forEach(([k, fill, ink]) => {
      const d = document.createElement("div");
      d.style.flexGrow = grow ? O[k] : 0.0001; d.style.flexBasis = "0";
      d.style.background = fill; d.style.color = ink;
      const share = O[k] / total; d.textContent = share > 0.08 ? Math.round(share * 100) + "%" : "";
      d.title = `${k}: ${nf(O[k])} (${pct(share)})`;
      bar.appendChild(d);
      const s = document.createElement("span"); const kk = document.createElement("span"); kk.className = "dotkey"; kk.style.borderRadius = "3px"; kk.style.background = fill;
      s.appendChild(kk); s.appendChild(document.createTextNode(`${k} · ${nf(O[k])}`)); leg.appendChild(s);
    });
  }
  build(reduce);
  whenShown(bar, () => requestAnimationFrame(() => [...bar.children].forEach((d, i) => d.style.flexGrow = O[parts[i][0]])));
  return { redraw: () => build(true) };
})();

/* ---------------------------------------------------------------- importance */
const importance = (function () {
  const svg = document.getElementById("impSvg"), I = Object.entries(DS.importance);
  const max = Math.max(...I.map(([, v]) => v.drop));
  const X0 = 270, X1 = 680, sc = v => Math.max(0, v) / max * (X1 - X0);
  let prog = 0;
  function draw(p) {
    clear(svg);
    const g = el("g", { class: "grid" }, svg);
    [0, 0.01, 0.02, 0.03].filter(v => v <= max * 1.05).forEach(v => { el("line", { x1: X0 + sc(v), y1: 6, x2: X0 + sc(v), y2: 352 }, g); txt(svg, X0 + sc(v), 372, v.toFixed(2), "lbl-muted", { "text-anchor": "middle" }); });
    I.forEach(([name, v], i) => {
      const y = 12 + i * 34, w = sc(v.drop) * p, h = 20;
      txt(svg, X0 - 14, y + 15, name, i < 3 ? "lbl-strong" : "lbl", { "text-anchor": "end" });
      if (w > 1) el("path", { d: `M${X0} ${y} h${Math.max(0, w - 4)} q4 0 4 4 v${h - 8} q0 4 -4 4 h${-Math.max(0, w - 4)} z`, fill: css("--s1"), opacity: i < 3 ? 1 : .6 }, svg);
      txt(svg, X0 + w + 8, y + 15, Math.abs(v.drop) < 0.0005 ? "0.000" : (v.drop >= 0 ? "−" : "+") + Math.abs(v.drop).toFixed(3), "lbl-muted", { opacity: p });
      const hit = el("rect", { x: 0, y: y - 6, width: 760, height: 32, class: "hit" }, svg);
      attachTip(hit, () => ({ title: name, rows: [{ value: "−" + v.drop.toFixed(4), label: "AUROC when shuffled", color: css("--s1") }, { value: "±" + v.sd.toFixed(4), label: "sd over 5 shuffles" }] }));
    });
  }
  draw(reduce ? 1 : 0);
  whenShown(svg, () => {
    if (reduce) return draw(1);
    const t0 = performance.now();
    (function f(now) { prog = ease(Math.min(1, (now - t0) / 1200)); draw(prog); if (prog < 1) requestAnimationFrame(f); })(t0);
  });
  return { redraw: () => draw(reduce ? 1 : Math.max(prog, 1)) };
})();

/* ---------------------------------------------------------------- per module */
const perModule = (function () {
  const svg = document.getElementById("modSvg"), M2 = Object.entries(DS.per_module).sort((a, b) => b[1].auc - a[1].auc);
  const lo = 0.6, hi = 0.78, X0 = 120, X1 = 620;
  const x = v => X0 + (v - lo) / (hi - lo) * (X1 - X0);
  const all = S.day14.later_run.boosted.auc;
  let prog = 0;
  function draw(p) {
    clear(svg);
    const g = el("g", { class: "grid" }, svg);
    [0.6, 0.65, 0.7, 0.75].forEach(v => { el("line", { x1: x(v), y1: 10, x2: x(v), y2: 232 }, g); txt(svg, x(v), 252, v.toFixed(2), "lbl-muted", { "text-anchor": "middle" }); });
    el("line", { x1: x(all), y1: 4, x2: x(all), y2: 232, stroke: css("--ink2"), opacity: .6 }, svg);
    txt(svg, x(all) + 6, 12, "all " + f3(all), "lbl-muted");
    M2.forEach(([m, v], i) => {
      const y = 30 + i * 36, vx = lerp(x(lo), x(v.auc), p);
      txt(svg, X0 - 14, y + 5, "Module " + m, "lbl-strong", { "text-anchor": "end" });
      el("line", { x1: x(lo), y1: y, x2: vx, y2: y, stroke: css("--axis"), "stroke-width": 2 }, svg);
      el("circle", { cx: vx, cy: y, r: 6.5, fill: css("--s1"), stroke: css("--surface"), "stroke-width": 2 }, svg);
      txt(svg, x(hi) + 14, y + 5, `${f3(v.auc)} · ${nf(v.n)} students`, "lbl", { opacity: p, style: "font-size:12.5px" });
      const hit = el("rect", { x: 0, y: y - 16, width: 760, height: 32, class: "hit" }, svg);
      attachTip(hit, () => ({ title: "Module " + m, rows: [{ value: f3(v.auc), label: "later-run AUROC, day 14", color: css("--s1") }, { value: nf(v.n), label: "test students" }, { value: pct(v.pass_rate), label: "passed" }] }));
    });
  }
  draw(reduce ? 1 : 0);
  whenShown(svg, () => {
    if (reduce) return draw(1);
    const t0 = performance.now();
    (function f(now) { prog = ease(Math.min(1, (now - t0) / 1200)); draw(prog); if (prog < 1) requestAnimationFrame(f); })(t0);
  });
  return { redraw: () => draw(reduce ? 1 : Math.max(prog, 1)) };
})();

function redrawAll() { [scrolly, clickLines, clickBars, withdraw, outcomes, importance, perModule, split, pipe, timeline, dumbbell, diverging, groups, slope, explorer].forEach(c => c.redraw()); }
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
