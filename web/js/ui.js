// Minimal panel helpers: collapsible sections with standard form controls.

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

let uid = 0;
const nextId = () => `c${++uid}`;

// A layer section: [checkbox] Title ▾  — checkbox shows/hides the layer, the title folds the body.
export function section(panel, title, { checked = null, open = false, onToggle, hint } = {}) {
  const sec = el('section', 'sec' + (open ? ' open' : ''));
  const head = el('div', 'sec-head');
  let box = null;
  if (checked !== null) {
    box = el('input');
    box.type = 'checkbox';
    box.checked = checked;
    box.title = `Show ${title.toLowerCase()}`;
    box.addEventListener('change', () => onToggle(box.checked));
    head.append(box);
  }
  const btn = el('button', 'sec-title');
  btn.type = 'button';
  btn.append(el('span', '', title), el('span', 'chev', '▸'));
  btn.setAttribute('aria-expanded', String(open));
  btn.addEventListener('click', () => {
    const o = sec.classList.toggle('open');
    btn.setAttribute('aria-expanded', String(o));
  });
  head.append(btn);
  const body = el('div', 'sec-body');
  if (hint) body.append(el('p', 'hint', hint));
  sec.append(head, body);
  panel.append(sec);
  return { sec, body, box, setChecked: (v) => { if (box) box.checked = v; } };
}

export function select(parent, label, options, value, onChange) {
  const row = el('label', 'row');
  row.append(el('span', 'lbl', label));
  const s = el('select');
  let group = null;
  for (const o of options) {
    if (o.group !== undefined && (!group || group.label !== o.group)) {
      group = el('optgroup');
      group.label = o.group;
      s.append(group);
    }
    const opt = el('option', '', o.label);
    opt.value = o.value;
    (o.group !== undefined ? group : s).append(opt);
  }
  s.value = value;
  s.addEventListener('change', () => onChange(s.value));
  row.append(s);
  parent.append(row);
  return { row, input: s, set: (v) => { s.value = v; } };
}

export function slider(parent, label, { min, max, step = 0.01, value, format = (v) => v, onInput }) {
  const row = el('label', 'row');
  const id = nextId();
  row.htmlFor = id;
  const lbl = el('span', 'lbl', label);
  const out = el('output', 'val', format(value));
  const r = el('input');
  r.type = 'range';
  r.id = id;
  Object.assign(r, { min, max, step, value });
  r.addEventListener('input', () => {
    const v = parseFloat(r.value);
    out.textContent = format(v);
    onInput(v);
  });
  row.append(lbl, r, out);
  parent.append(row);
  return {
    row, input: r,
    set: (v) => { r.value = v; out.textContent = format(v); },
    setEnabled: (on) => { r.disabled = !on; row.classList.toggle('disabled', !on); },
  };
}

export function checkbox(parent, label, checked, onChange, { swatch } = {}) {
  const row = el('label', 'check');
  const c = el('input');
  c.type = 'checkbox';
  c.checked = checked;
  c.addEventListener('change', () => onChange(c.checked));
  row.append(c);
  if (swatch) {
    const sw = el('span', 'swatch');
    sw.style.background = swatch;
    row.append(sw);
  }
  row.append(el('span', '', label));
  parent.append(row);
  return { row, input: c, set: (v) => { c.checked = v; } };
}

export function inline(parent, cls = 'inline') {
  const d = el('div', cls);
  parent.append(d);
  return d;
}

export function buttons(parent, list) {
  const d = el('div', 'btns');
  for (const b of list) {
    const x = el('button', '', b.label);
    x.type = 'button';
    if (b.title) x.title = b.title;
    x.addEventListener('click', b.onClick);
    d.append(x);
  }
  parent.append(d);
  return d;
}

export function note(parent, text, cls = 'note') {
  const p = el('p', cls, text);
  parent.append(p);
  return p;
}

// Foldable list of rows grouped under small headings.
export function groupedList(parent, title, groups) {
  const det = el('details', 'list');
  det.append(el('summary', '', title));
  const box = el('div', 'list-body');
  for (const [heading, rows] of groups) {
    if (heading) box.append(el('div', 'list-head', heading));
    for (const r of rows) checkbox(box, r.label, r.checked, r.onChange, { swatch: r.swatch });
  }
  det.append(box);
  parent.append(det);
  return det;
}

export { el };
