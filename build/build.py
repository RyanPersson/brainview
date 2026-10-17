#!/usr/bin/env python3
"""Convert the downloaded open datasets (raw/) into compact binaries for the web viewer (web/data/).

Usage:  .venv/bin/python -I build/build.py [cortex] [atlases] [meshes] [tracts] [flow] [ftract] [volumes] [tms] [check]
        (no args = everything; atlases is the parcellation part of cortex, tms needs the meshes)

Every output array is written as  web/data/<name>.bin.gz  (raw little-endian bytes, gzipped;
the viewer inflates them itself, so any static host works).  web/data/manifest.json
describes all of them.  Common frame: MNI152NLin2009cAsym world coordinates (RAS, mm).
"""
import gzip
import json
import os
import re
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage, sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "web", "data")
TF_FS = os.path.join(RAW, "templateflow", "tpl-fsaverage", "tpl-fsaverage_")
TF_MNI = os.path.join(RAW, "templateflow", "tpl-MNI152NLin2009cAsym", "tpl-MNI152NLin2009cAsym_")
FS_LUT = os.path.join(ROOT, ".venv", "lib", "python3.12", "site-packages", "mne", "data", "FreeSurferColorLUT.txt")

# FreeSurfer's published MNI152 -> MNI305 affine (fsaverage surfaces live in MNI305).
MNI152_TO_305 = np.array([
    [1.0022, 0.0071, -0.0177, 0.0528],
    [-0.0146, 0.9990, 0.0027, -1.5519],
    [0.0129, 0.0094, 1.0027, -1.2012],
    [0, 0, 0, 1],
])
MNI305_TO_152 = np.linalg.inv(MNI152_TO_305)

MANIFEST_PATH = os.path.join(OUT, "manifest.json")


# ----------------------------------------------------------------------------- helpers

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name, arr):
    arr = np.ascontiguousarray(arr)
    data = arr.tobytes()
    path = os.path.join(OUT, name + ".bin.gz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # leave an unchanged file alone, and write no timestamp, so a rebuild doesn't churn the committed data
    if not (os.path.exists(path) and gzip.decompress(open(path, "rb").read()) == data):
        with open(path, "wb") as fh, gzip.GzipFile("", "wb", 9, fh, mtime=0) as f:
            f.write(data)
    return {"url": "data/" + name + ".bin.gz", "dtype": arr.dtype.name, "length": int(arr.size)}


def save_quantized(name, v, dtype=np.uint16):
    """Float Nx3 -> uint16 per-axis quantized.  decode: x = min + q * scale."""
    v = np.asarray(v, np.float64)
    lo = v.min(0)
    hi = v.max(0)
    scale = np.maximum(hi - lo, 1e-6) / 65535.0
    q = np.round((v - lo) / scale).astype(dtype)
    d = save(name, q)
    d.update(min=lo.round(5).tolist(), scale=scale.tolist())
    return d


def apply_affine(m, v):
    return v @ m[:3, :3].T + m[:3, 3]


def signed_volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


def orient_outward(v, f):
    return f if signed_volume(v, f) >= 0 else f[:, ::-1].copy()


def surface_area(v, f):
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()


def mesh_adjacency(f, n):
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    a = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    deg = np.asarray(a.sum(1)).ravel()
    return sparse.diags(1.0 / deg) @ a


def taubin(v, f, iters=10, lam=0.5, mu=-0.53):
    w = mesh_adjacency(f, len(v))
    x = v.copy()
    for _ in range(iters):
        x = x + lam * (w @ x - x)
        x = x + mu * (w @ x - x)
    return x


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as fh:
            return json.load(fh)
    return {}


def hexcol(rgb):
    r, g, b = [int(round(c)) for c in rgb[:3]]
    return "#%02x%02x%02x" % (r, g, b)


# ----------------------------------------------------------------------------- names

DK_WORDS = ["banks", "sts", "caudal", "anterior", "cingulate", "middle", "frontal", "cuneus", "entorhinal",
            "fusiform", "inferior", "parietal", "temporal", "isthmus", "lateral", "occipital", "orbitofrontal",
            "lingual", "medial", "parahippocampal", "paracentral", "pars", "opercularis", "orbitalis",
            "triangularis", "pericalcarine", "postcentral", "posterior", "precentral", "precuneus", "rostral",
            "superior", "supramarginal", "pole", "transverse", "insula", "corpus", "callosum", "unknown"]


def split_words(s):
    out, i = [], 0
    while i < len(s):
        for w in sorted(DK_WORDS, key=len, reverse=True):
            if s.startswith(w, i):
                out.append(w)
                i += len(w)
                break
        else:
            out.append(s[i:])
            break
    return out


def pretty_dk(name):
    m = re.match(r"^([a-z]+)(?:_(\d+))?$", name)
    if not m:
        return name
    words = split_words(m.group(1))
    s = " ".join(words)
    s = s.replace("banks sts", "banks of the superior temporal sulcus")
    s = s[:1].upper() + s[1:]
    return s + (f" · {m.group(2)}" if m.group(2) else "")


DESTRIEUX_TOKENS = {
    "G": "gyrus", "S": "sulcus", "G_and_S": "gyrus & sulcus", "Lat_Fis": "lateral fissure", "Pole": "pole",
    "front": "frontal", "inf": "inferior", "sup": "superior", "med": "medial", "lat": "lateral", "ant": "anterior",
    "post": "posterior", "pariet": "parietal", "oc": "occipital", "occipital": "occipital", "temp": "temporal",
    "temporal": "temporal", "cingul": "cingulate", "Ant": "anterior", "Mid": "middle", "Post": "posterior",
    "dorsal": "dorsal", "ventral": "ventral", "orbital": "orbital", "subcallosal": "subcallosal",
    "transv": "transverse", "transverse": "transverse", "Horizont": "horizontal ramus", "Vertical": "vertical ramus",
    "middle": "middle", "circular": "circular", "insula": "insula", "intrapariet": "intraparietal",
    "and": "&", "P": "parietal", "calcarine": "calcarine", "callosal": "callosal", "central": "central",
    "collat": "collateral", "cuneus": "cuneus", "rectus": "rectus", "short": "short", "long": "long",
    "angular": "angular", "supramar": "supramarginal", "precentral": "precentral", "postcentral": "postcentral",
    "precuneus": "precuneus", "paracentral": "paracentral", "parahip": "parahippocampal", "fusifor": "fusiform",
    "lingual": "lingual", "marginalis": "marginal", "Plan": "planum", "polar": "polare", "tempo": "temporale",
    "Opercular": "opercular part", "Orbital": "orbital part", "Triangul": "triangular part",
    "frontomargin": "frontomarginal", "transv_frontopol": "transverse frontopolar", "pericallosal": "pericallosal",
    "oc-temp": "occipito-temporal", "orbital-H_Shaped": "orbital H-shaped", "suborbital": "suborbital",
    "subparietal": "subparietal", "interm": "intermediate", "prim-Jensen": "primary (Jensen)",
    "paracentral": "paracentral", "MedWall": "medial wall", "Medial_wall": "medial wall", "Unknown": "unknown",
}


def pretty_destrieux(name):
    if name in ("Unknown", "Medial_wall"):
        return "Medial wall"
    toks = re.split(r"[_\-]", name)
    words = [DESTRIEUX_TOKENS.get(t, t) for t in toks]
    s = " ".join(words).replace("gyrus & sulcus", "gyrus & sulcus")
    return s[:1].upper() + s[1:] + f"  ({name})"


YEO7 = {"Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "Dorsal attention",
        "SalVentAttn": "Salience / ventral attention", "Limbic": "Limbic", "Cont": "Frontoparietal control",
        "Default": "Default mode"}
YEO17 = {"VisCent": "Visual A (central)", "VisPeri": "Visual B (peripheral)", "SomMotA": "Somatomotor A",
         "SomMotB": "Somatomotor B", "DorsAttnA": "Dorsal attention A", "DorsAttnB": "Dorsal attention B",
         "SalVentAttnA": "Salience / ventral attention A", "SalVentAttnB": "Salience / ventral attention B",
         "LimbicA": "Limbic A (temporal pole)", "LimbicB": "Limbic B (orbitofrontal)", "ContA": "Control A",
         "ContB": "Control B", "ContC": "Control C", "DefaultA": "Default A", "DefaultB": "Default B",
         "DefaultC": "Default C", "TempPar": "Temporo-parietal"}


def pretty_schaefer(name):
    m = re.match(r"^(7|17)Networks_[LR]H_([A-Za-z]+)_?(.*)$", name)
    if not m:
        return "Medial wall" if "Medial_Wall" in name else name
    net = (YEO7 if m.group(1) == "7" else YEO17).get(m.group(2), m.group(2))
    rest = m.group(3).replace("_", " ")
    return f"{net} · {rest}" if rest else net


# ----------------------------------------------------------------------------- cortex

ATLASES = [
    # id, label, file key (or special), description
    ("dk", "Desikan–Killiany", "atlas-Desikan2006_seg-aparc_desc-curated",
     "34 gyral regions per hemisphere (Desikan et al. 2006) — the classic FreeSurfer aparc"),
    ("destrieux", "Destrieux", "atlas-Destrieux2009",
     "74 gyri and sulci per hemisphere (Destrieux et al. 2010) — FreeSurfer aparc.a2009s"),
    ("hcpmmp", "Glasser HCP-MMP1", "hcpmmp",
     "180 areas per hemisphere from multimodal MRI (Glasser et al. 2016)"),
    ("brainnetome", "Brainnetome", "atlas-brainnetome",
     "105 cortical areas per hemisphere from connectivity (Fan et al. 2016)"),
    ("yeo7", "Yeo 7 networks", "atlas-Yeo2011_seg-7n",
     "7 resting-state functional networks (Yeo et al. 2011)"),
    ("yeo17", "Yeo 17 networks", "atlas-Yeo2011_seg-17n",
     "17 resting-state functional networks (Yeo et al. 2011)"),
    ("schaefer100", "Schaefer 100", "atlas-Schaefer2018_seg-7n_scale-100",
     "100 functional parcels in 7 networks (Schaefer et al. 2018)"),
    ("schaefer400", "Schaefer 400", "atlas-Schaefer2018_seg-7n_scale-400",
     "400 functional parcels in 7 networks (Schaefer et al. 2018)"),
    ("schaefer1000", "Schaefer 1000", "atlas-Schaefer2018_seg-17n_scale-1000",
     "1000 functional parcels in 17 networks (Schaefer et al. 2018)"),
    ("lausanne1", "Lausanne scale 1", "atlas-Lausanne2018_scale-1",
     "Multi-scale Lausanne 2018 parcellation, coarsest (≈ Desikan–Killiany)"),
    ("lausanne3", "Lausanne scale 3", "atlas-Lausanne2018_scale-3",
     "Lausanne 2018, scale 3 (~110 regions per hemisphere)"),
    ("lausanne5", "Lausanne scale 5", "atlas-Lausanne2018_scale-5",
     "Lausanne 2018, finest scale (~500 regions per hemisphere)"),
    ("bodymap", "Body map (motor & touch strips)", "bodymap",
     "Which body part each stretch of the motor strip (primary motor cortex, darker) and the touch strip "
     "(primary somatosensory cortex, lighter) serves, from the body-part subregions of the Brainnetome atlas "
     "(Fan et al. 2016). This is the order of Penfield's homunculus, though the borders are not sharp. "
     "Gordon et al. (2023) also found zones between the body parts that plan whole-body actions."),
]

# Brainnetome subregions of the primary motor (area 4) and somatosensory (areas 1/2/3) strips, by body part.
# Touch shares the movement colour, mixed toward white.
BODYMAP = [
    # Brainnetome name, body part, strip, colour
    ("a4ll", "Leg & foot", "movement", "#3d7bd9"),
    ("a4t", "Trunk", "movement", "#2bb3a0"),
    ("a4ul", "Arm & hand", "movement", "#e3b22b"),
    ("a4hf", "Head & face", "movement", "#e0503c"),
    ("a4tl", "Tongue & throat", "movement", "#b05cd6"),
    ("a1/2/3ll", "Leg & foot", "touch", "#3d7bd9"),
    ("a1/2/3tru", "Trunk", "touch", "#2bb3a0"),
    ("a1/2/3ulhf", "Arm, hand & face", "touch", "#e88a33"),
    ("a1/2/3tonia", "Tongue & throat", "touch", "#b05cd6"),
]
STRIPS = {"movement": "movement · primary motor cortex", "touch": "touch · primary somatosensory cortex"}


def body_map(hemi):
    """Body-map labels on fsaverage from Brainnetome, plus one label anchor per motor body part: the vertex on
    a gyral crown nearest the centre of that part's crown, so the label sits on visible cortex."""
    bn, names, _ = read_atlas(hemi, "brainnetome", "atlas-brainnetome")
    pial = apply_affine(MNI305_TO_152, nib.load(f"{TF_FS}hemi-{hemi}_den-164k_pial.surf.gii").agg_data()[0])
    sulc = nib.load(f"{TF_FS}hemi-{hemi}_den-164k_sulc.shape.gii").agg_data()
    comp = np.zeros(len(bn), np.uint16)
    out_names, out_colors, anchors = ["Rest of the cortex"], ["#8c8c8c"], []
    for k, (region, part, strip, col) in enumerate(BODYMAP, start=1):
        m = bn == names.index(region)
        comp[m] = k
        rgb = np.array([int(col[i:i + 2], 16) for i in (1, 3, 5)], float)
        if strip == "touch":
            rgb = rgb + (255 - rgb) * 0.5
        out_names.append(f"{part} ({STRIPS[strip]})")
        out_colors.append(hexcol(rgb))
        if strip == "movement":
            crown = np.nonzero(m & (sulc < 0))[0]
            c = pial[crown].mean(0)
            anchors.append({"text": part, "color": col,
                            "vertex": int(crown[np.argmin(((pial[crown] - c) ** 2).sum(1))])})
    return comp, out_names, out_colors, anchors


def atlas_pretty(aid, name):
    if aid in ("dk",) or aid.startswith("lausanne"):
        return pretty_dk(name)
    if aid == "destrieux":
        return pretty_destrieux(name)
    if aid.startswith("schaefer"):
        return pretty_schaefer(name)
    if aid == "hcpmmp":
        m = re.match(r"^[LR]_(.*)_ROI$", name)
        return ("Area " + m.group(1)) if m else ("Medial wall" if name == "???" else name)
    if aid == "brainnetome":
        return "Medial wall" if "unknown" in name else name
    return name


def read_atlas(hemi, aid, key):
    """-> (per-vertex compact index uint16, names[], colors[])  index 0 is always 'unknown/medial wall'."""
    if key == "hcpmmp":
        labels, ctab, names = nib.freesurfer.read_annot(os.path.join(RAW, "hcpmmp", f"{hemi.lower()}h.HCPMMP1.annot"))
        names = [n.decode() for n in names]
        keys = list(range(len(names)))
        colors = [hexcol(c[:3]) for c in ctab]
        labels = np.where(labels < 0, 0, labels)
    else:
        g = nib.load(f"{TF_FS}hemi-{hemi}_den-164k_{key}_dseg.label.gii")
        labels = g.agg_data().astype(np.int64)
        lt = g.labeltable.labels
        keys = [l.key for l in lt]
        names = [l.label for l in lt]
        colors = [hexcol(np.array(l.rgba[:3]) * 255) for l in lt]
    remap = {k: i for i, k in enumerate(keys)}
    idx = np.array([remap.get(int(k), 0) for k in range(int(labels.max()) + 1)], dtype=np.int64)
    # some files mark the medial wall with -1; negative indices would wrap to the last label
    comp = np.where(labels >= 0, idx[np.maximum(labels, 0)], 0)
    return comp.astype(np.uint16), names, colors


def yeo_names_from_schaefer(hemi, which):
    """Name Yeo network indices by majority vote of the overlapping Schaefer parcel names."""
    ylab, yn, _ = read_atlas(hemi, "y", f"atlas-Yeo2011_seg-{which}")
    key = "atlas-Schaefer2018_seg-7n_scale-400" if which == "7n" else "atlas-Schaefer2018_seg-17n_scale-1000"
    slab, sn, _ = read_atlas(hemi, "s", key)
    out = []
    table = YEO7 if which == "7n" else YEO17
    for i in range(len(yn)):
        m = ylab == i
        if i == 0 or not m.any():
            out.append("Medial wall")
            continue
        votes = {}
        for s in slab[m]:
            mm = re.match(r"^(?:7|17)Networks_[LR]H_([A-Za-z]+)", sn[s])
            if mm:
                votes[mm.group(1)] = votes.get(mm.group(1), 0) + 1
        best = max(votes, key=votes.get) if votes else None
        out.append(f"{table.get(best, best)} (network {i})")
    return out


def inflate(v, f, keep, lam=0.5):
    w = mesh_adjacency(f, len(v))
    a0 = surface_area(v, f)
    c0 = v.mean(0)
    x = v.copy()
    res = {}
    for k in range(1, max(keep) + 1):
        x = x + lam * (w @ x - x)
        if k % 5 == 0 or k in keep:
            c = x.mean(0)
            x = c0 + (x - c) * np.sqrt(a0 / surface_area(x, f))
        if k in keep:
            res[k] = x.copy()
    return res


def build_cortex(man):
    log("cortex: loading fsaverage 164k")
    hemis = {}
    for h in "LR":
        surf = {}
        for s in ["pial", "white", "sphere"]:
            v, f = nib.load(f"{TF_FS}hemi-{h}_den-164k_{s}.surf.gii").agg_data()
            surf[s] = v.astype(np.float64)
            faces = f.astype(np.int64)
        hemis[h] = (surf, faces)

    out = {"surfaces": [], "hemis": {}, "scalars": [], "atlases": []}
    INFL = [300, 2000]
    SURFS = [("pial", "Pial"), ("white", "White"), ("inflated", "Inflated"),
             ("veryinflated", "Very inflated"), ("sphere", "Sphere")]
    out["surfaces"] = [{"id": s, "label": l} for s, l in SURFS]

    # anatomical surfaces in MNI152
    geo = {}
    for h in "LR":
        surf, faces = hemis[h]
        pial = apply_affine(MNI305_TO_152, surf["pial"])
        white = apply_affine(MNI305_TO_152, surf["white"])
        faces = orient_outward(pial, faces)
        log(f"cortex {h}: inflating")
        infl = inflate(white, faces, INFL)
        geo[h] = dict(pial=pial, white=white, inflated=infl[INFL[0]], veryinflated=infl[INFL[1]],
                      sphere=surf["sphere"], faces=faces)

    # place the non-anatomical shapes so the hemispheres sit side by side instead of interpenetrating
    gap = 4.0
    for key in ["inflated", "veryinflated"]:
        lx = geo["L"][key][:, 0].max()
        rx = geo["R"][key][:, 0].min()
        geo["L"][key] = geo["L"][key] - [lx + gap / 2, 0, 0]
        geo["R"][key] = geo["R"][key] - [rx - gap / 2, 0, 0]
    radius = 62.0
    for h, sgn in (("L", -1), ("R", 1)):
        s = geo[h]["sphere"]
        s = s / np.linalg.norm(s, axis=1, keepdims=True) * radius
        c = geo[h]["pial"].mean(0)
        geo[h]["sphere"] = s + [sgn * (radius + gap / 2), c[1], c[2]]

    same_faces = np.array_equal(geo["L"]["faces"], geo["R"]["faces"])
    log("cortex: L/R share faces:", same_faces)
    for h in "LR":
        g = geo[h]
        if h == "R" and same_faces:
            fentry = out["hemis"]["L"]["faces"]
        else:
            fentry = save(f"cortex/{h}_faces", g["faces"].astype(np.uint32))
        hd = {"vertices": int(len(g["pial"])), "faces": fentry}
        for s, _ in SURFS:
            hd[s] = save_quantized(f"cortex/{h}_{s}", g[s])
        out["hemis"][h] = hd

    # scalar maps (uint8 quantized over a robust range shared by both hemispheres)
    scal = {}
    for h in "LR":
        scal.setdefault("sulc", []).append(nib.load(f"{TF_FS}hemi-{h}_den-164k_sulc.shape.gii").agg_data())
        scal.setdefault("curv", []).append(nib.load(f"{TF_FS}hemi-{h}_den-164k_curv.shape.gii").agg_data())
        surf, _ = hemis[h]
        scal.setdefault("thickness", []).append(np.linalg.norm(surf["pial"] - surf["white"], axis=1))
    SC = [("sulc", "Sulcal depth", "Depth below the average brain hull (FreeSurfer sulc); deep sulci positive", "diverging"),
          ("curv", "Curvature", "Mean curvature of the white surface; sulci positive, gyri negative", "diverging"),
          ("thickness", "Pial–white distance", "Distance between pial and white surfaces of the average brain (mm)", "sequential")]
    for sid, label, desc, cmap in SC:
        allv = np.concatenate(scal[sid])
        if cmap == "diverging":
            m = float(np.percentile(np.abs(allv), 99))
            lo, hi = -m, m
        else:
            lo, hi = float(np.percentile(allv, 0.5)), float(np.percentile(allv, 99.5))
        files = {}
        for h, arr in zip("LR", scal[sid]):
            q = np.clip(np.round((arr - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
            files[h] = save(f"cortex/{h}_{sid}", q)
        out["scalars"].append({"id": sid, "label": label, "description": desc, "cmap": cmap,
                               "min": lo, "max": hi, "files": files})
        if sid == "sulc":
            out["shading"] = sid

    man["cortex"] = out
    build_atlases(man)


def build_atlases(man):
    """The cortical parcellations (also a step of its own, as it doesn't need the surfaces rebuilt)."""
    out = man["cortex"]
    out["atlases"] = []
    for aid, label, key, desc in ATLASES:
        entry = {"id": aid, "label": label, "description": desc, "hemis": {}}
        for h in "LR":
            if aid == "bodymap":
                lab, names, colors, anchors = body_map(h)
                entry["hemis"][h] = {"labels": save(f"cortex/{h}_atlas_{aid}", lab), "names": names,
                                     "colors": colors, "anchors": anchors}
                entry["legend"] = [{"name": a["text"], "color": a["color"]} for a in anchors]
                continue
            lab, names, colors = read_atlas(h, aid, key)
            if aid == "yeo7":
                names = yeo_names_from_schaefer(h, "7n")
            elif aid == "yeo17":
                names = yeo_names_from_schaefer(h, "17n")
            else:
                names = [atlas_pretty(aid, n) for n in names]
            names[0] = "Medial wall / unlabelled"
            colors[0] = "#8c8c8c"
            entry["hemis"][h] = {"labels": save(f"cortex/{h}_atlas_{aid}", lab), "names": names, "colors": colors}
        n = len(entry["hemis"]["L"]["names"]) - 1
        log(f"atlas {aid}: {n} labels/hemi")
        out["atlases"].append(entry)


# ----------------------------------------------------------------------------- volumetric meshes

def read_fs_lut():
    lut = {}
    with open(FS_LUT) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 5 and p[0].isdigit():
                lut[int(p[0])] = (p[1], [int(p[2]), int(p[3]), int(p[4])])
    return lut


def mask_to_mesh(mask, affine, sigma=1.0, upsample=1, target_faces=None, smooth=5):
    from skimage import measure
    idx = np.argwhere(mask)
    lo = idx.min(0) - 4
    hi = idx.max(0) + 5
    # a 4-voxel empty margin all round, also where the mask runs into the edge of the volume, so the
    # surface is closed there
    sub = np.pad(mask, 4)[lo[0] + 4:hi[0] + 4, lo[1] + 4:hi[1] + 4, lo[2] + 4:hi[2] + 4].astype(np.float32)
    n_in = np.array(sub.shape)
    if upsample != 1:
        sub = ndimage.zoom(sub, upsample, order=1)
    sub = ndimage.gaussian_filter(sub, sigma * upsample)
    level = 0.5 if sub.max() > 0.5 else sub.max() * 0.5
    verts, faces, _, _ = measure.marching_cubes(sub, level)
    n_out = np.array(sub.shape)
    verts = verts * (n_in - 1) / (n_out - 1) + lo  # back to voxel indices of the full volume
    v = apply_affine(affine, verts)
    f = faces.astype(np.int64)
    if target_faces and len(f) > target_faces:
        import fast_simplification
        v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32),
                                            target_reduction=1 - target_faces / len(f))
        v = v.astype(np.float64)
        f = f.astype(np.int64)
    if smooth:
        v = taubin(v, f, smooth)
    f = orient_outward(v, f)
    return v, f


def build_mesh_group(gid, label, desc, structures, man):
    """structures: list of dicts(name, mask, color, affine, opts)"""
    allv, allf, items = [], [], []
    vofs = fofs = 0
    for s in structures:
        t0 = time.time()
        v, f = mask_to_mesh(s["mask"], s["affine"], **s.get("opts", {}))
        items.append({"id": s["id"], "name": s["name"], "color": s["color"], "hemi": s.get("hemi", ""),
                      "category": s.get("category", ""), "visible": s.get("visible", True),
                      "vStart": vofs, "vCount": len(v), "iStart": fofs * 3, "iCount": len(f) * 3,
                      "voxels": int(s["mask"].sum())})
        allv.append(v)
        allf.append(f + vofs)
        vofs += len(v)
        fofs += len(f)
        log(f"  {gid}/{s['name']}: {len(v)} verts {len(f)} faces ({time.time() - t0:.1f}s)")
    v = np.concatenate(allv)
    f = np.concatenate(allf)
    man.setdefault("meshGroups", [])
    man["meshGroups"] = [g for g in man["meshGroups"] if g["id"] != gid]
    man["meshGroups"].append({"id": gid, "label": label, "description": desc,
                              "positions": save_quantized(f"meshes/{gid}_pos", v),
                              "indices": save(f"meshes/{gid}_idx", f.astype(np.uint32)),
                              "structures": items})


ASEG = [
    # labels, name, category, default visible, opts
    ([10, 49], "Thalamus", "Diencephalon", True),
    ([11, 50], "Caudate nucleus", "Basal ganglia", True),
    ([12, 51], "Putamen", "Basal ganglia", True),
    ([13, 52], "Globus pallidus", "Basal ganglia", True),
    ([26, 58], "Nucleus accumbens", "Basal ganglia", True),
    ([17, 53], "Hippocampus", "Limbic", True),
    ([18, 54], "Amygdala", "Limbic", True),
    ([28, 60], "Ventral diencephalon", "Diencephalon", True),
    ([16], "Brainstem", "Brainstem & cerebellum", True),
    ([8, 47], "Cerebellar cortex", "Brainstem & cerebellum", True),
    ([7, 46], "Cerebellar white matter", "Brainstem & cerebellum", False),
    ([4, 43], "Lateral ventricle", "Ventricles", True),
    ([5, 44], "Inferior lateral ventricle", "Ventricles", True),
    ([14], "Third ventricle", "Ventricles", True),
    ([15], "Fourth ventricle", "Ventricles", True),
    ([31, 63], "Choroid plexus", "Ventricles", False),
    ([251, 252, 253, 254, 255], "Corpus callosum", "White matter", False),
    ([85], "Optic chiasm", "White matter", True),
    ([2, 41], "Cerebral white matter", "Cerebrum", False),
    ([3, 42], "Cerebral cortex (volumetric)", "Cerebrum", False),
]


def build_meshes(man):
    lut = read_fs_lut()
    img = nib.load(TF_MNI + "res-01_seg-aseg_dseg.nii.gz")
    aseg = np.asarray(img.dataobj).astype(np.int32)
    aff = img.affine
    structs = []
    log("meshes: aseg")
    for labels, name, cat, vis in ASEG:
        big = name.startswith("Cerebral") or name.startswith("Cerebellar")
        per_side = len(labels) == 2
        groups = [(labels[0:1], "L"), (labels[1:2], "R")] if per_side else [(labels, "")]
        for labs, h in groups:
            m = np.isin(aseg, labs)
            if m.sum() < 5:
                continue
            col = lut[labs[0]][1] if labs[0] not in (2, 41) else [236, 232, 220]
            if labs[0] in (251, 252, 253, 254, 255):
                col = [70, 110, 230]
            opts = {"sigma": 0.8, "upsample": 1 if big or m.sum() > 20000 else 2,
                    "target_faces": 160000 if big else 40000, "smooth": 4}
            structs.append({"id": f"aseg_{labs[0]}", "name": name, "hemi": h, "category": cat, "visible": vis,
                            "mask": m, "affine": aff, "color": hexcol(col), "opts": opts})
    build_mesh_group("aseg", "Subcortical segmentation",
                     "FreeSurfer-style aseg labels of the MNI152 2009c template (TemplateFlow)", structs, man)

    log("meshes: MASSP nuclei")
    img = nib.load(TF_MNI + "res-01_atlas-MASSP20_dseg.nii.gz")
    massp = np.rint(np.asarray(img.dataobj)).astype(np.int32)
    rows = [l.rstrip("\n").split("\t") for l in open(TF_MNI + "res-01_atlas-MASSP20_dseg.tsv")][1:]
    structs = []
    for i, (idx, name) in enumerate((int(r[0]), r[1]) for r in rows if r and r[0].isdigit()):
        m = massp == idx
        if m.sum() < 8:
            continue
        h = "L" if name.startswith("left-") else "R" if name.startswith("right-") else ""
        base = re.sub(r"^(left|right)-", "", name).replace("-", " ")
        base = base[:1].upper() + base[1:]
        hue = (i * 0.61803398875) % 1.0
        import colorsys
        col = [c * 255 for c in colorsys.hls_to_rgb(hue, 0.58, 0.65)]
        structs.append({"id": f"massp_{idx}", "name": base, "hemi": h, "category": "MASSP", "mask": m,
                        "affine": img.affine, "color": hexcol(col),
                        "opts": {"sigma": 0.7, "upsample": 2, "target_faces": 30000, "smooth": 4}})
    build_mesh_group("massp", "Subcortical nuclei (MASSP)",
                     "Detailed subcortical nuclei & hippocampal subfields (MASSP atlas, Bazin et al. 2020)",
                     structs, man)

    log("meshes: thalamic nuclei")
    img = nib.load(TF_MNI + "res-01_atlas-MIAL67ThalamicNuclei_dseg.nii.gz")
    thal = np.asarray(img.dataobj).astype(np.int32)
    rows = [l.rstrip("\n").split("\t") for l in open(TF_MNI + "atlas-MIAL67ThalamicNuclei_dseg.tsv")][1:]
    structs = []
    for r in rows:
        if not r or not r[0].isdigit():
            continue
        idx, name, col = int(r[0]), r[1], r[2]
        m = thal == idx
        if m.sum() < 5:
            continue
        h = "L" if "-lh-" in name else "R"
        base = re.sub(r"^thal-[lr]h-", "", name).replace("-", " ")
        base = base[:1].upper() + base[1:]
        structs.append({"id": f"thal_{idx}", "name": base, "hemi": h, "category": "Thalamus", "mask": m,
                        "affine": img.affine, "color": col[:7],
                        "opts": {"sigma": 0.7, "upsample": 2, "target_faces": 30000, "smooth": 4}})
    build_mesh_group("thalamic", "Thalamic nuclei (MIAL)",
                     "Thalamic nuclei groups (Najdenovska et al. 2018, MIAL 67-subject atlas)", structs, man)

    log("meshes: context hulls")
    structs = []
    img = nib.load(TF_MNI + "res-01_desc-brain_mask.nii.gz")
    m = np.asarray(img.dataobj) > 0
    structs.append({"id": "hull_brain", "name": "Brain hull", "category": "Context", "mask": m, "visible": True,
                    "affine": img.affine, "color": "#9fb7d9",
                    "opts": {"sigma": 2.0, "upsample": 1, "target_faces": 60000, "smooth": 6}})
    # The head comes from thresholding the full-head T1. TemplateFlow's desc-head_mask can't be used: it
    # sits on a 545x653x545 grid and is solid below z = +5 mm, which made a huge flat sheet under a dome.
    img = nib.load(TF_MNI + "res-01_T1w.nii.gz")
    m = ndimage.binary_opening(np.asarray(img.dataobj) > 1000)  # the background tops out near 600
    lab, _ = ndimage.label(m)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    m = lab == sizes.argmax()
    # The template's box cuts the head at the brow (front) and below the cerebellum (bottom). Count those
    # cut faces as tissue, so the dark skull between the scalp and the brain is an enclosed cavity and gets
    # filled; interior cavities would only add hidden surfaces.
    p = np.pad(m, 1)
    p[:, -1, :] = True
    p[:, :, 0] = True
    m = ndimage.binary_fill_holes(p)[1:-1, 1:-1, 1:-1]
    structs.append({"id": "hull_head", "name": "Head (skin)", "category": "Context", "mask": m, "visible": False,
                    "affine": img.affine, "color": "#e3b9a0",
                    "opts": {"sigma": 1.2, "upsample": 1, "target_faces": 120000, "smooth": 4}})
    build_mesh_group("context", "Context surfaces",
                     "Outer hulls of the MNI152 2009c brain (brain mask) and head (skin surface of the T1)",
                     structs, man)


# ----------------------------------------------------------------------------- tracts

TRACTS = {
    # abbreviation: (full name, category)
    "AC": ("Anterior commissure", "Commissural"), "PC": ("Posterior commissure", "Commissural"),
    "CC": ("Corpus callosum", "Commissural"), "CCMid": ("Corpus callosum (body)", "Commissural"),
    "CC_ForcepsMajor": ("Forceps major", "Commissural"), "CC_ForcepsMinor": ("Forceps minor", "Commissural"),
    "AF": ("Arcuate fasciculus", "Association"), "AST": ("Frontal aslant tract", "Association"),
    "C": ("Cingulum", "Association"), "EMC": ("Extreme capsule", "Association"),
    "IFOF": ("Inferior fronto-occipital fasciculus", "Association"),
    "ILF": ("Inferior longitudinal fasciculus", "Association"),
    "MdLF": ("Middle longitudinal fasciculus", "Association"),
    "SLF": ("Superior longitudinal fasciculus", "Association"), "UF": ("Uncinate fasciculus", "Association"),
    "VOF": ("Vertical occipital fasciculus", "Association"),
    "AR": ("Acoustic radiation", "Projection"), "CS": ("Corticostriatal tract", "Projection"),
    "CST": ("Corticospinal tract", "Projection"), "CT": ("Corticothalamic tract", "Projection"),
    "FPT": ("Frontopontine tract", "Projection"), "OPT": ("Occipitopontine tract", "Projection"),
    "PPT": ("Parietopontine tract", "Projection"), "TPT": ("Temporopontine tract", "Projection"),
    "OR": ("Optic radiation", "Projection"), "F": ("Fornix", "Projection"),
    "CB": ("Cerebellum", "Cerebellar"), "V": ("Vermis", "Cerebellar"),
    "ICP": ("Inferior cerebellar peduncle", "Cerebellar"), "MCP": ("Middle cerebellar peduncle", "Cerebellar"),
    "SCP": ("Superior cerebellar peduncle", "Cerebellar"),
    "CTT": ("Central tegmental tract", "Brainstem"), "DLF": ("Dorsal longitudinal fasciculus", "Brainstem"),
    "LL": ("Lateral lemniscus", "Brainstem"), "ML": ("Medial lemniscus", "Brainstem"),
    "MLF": ("Medial longitudinal fasciculus", "Brainstem"), "RST": ("Reticulospinal tract", "Brainstem"),
    "STT": ("Spinothalamic tract", "Brainstem"),
    "CNII": ("Optic nerve (CN II)", "Cranial nerves"), "CNIII": ("Oculomotor nerve (CN III)", "Cranial nerves"),
    "CNIV": ("Trochlear nerve (CN IV)", "Cranial nerves"), "CNV": ("Trigeminal nerve (CN V)", "Cranial nerves"),
    "CNVII": ("Facial nerve (CN VII)", "Cranial nerves"),
    "CNVIII": ("Vestibulocochlear nerve (CN VIII)", "Cranial nerves"),
}
TRACT_CATS = ["Association", "Commissural", "Projection", "Cerebellar", "Brainstem", "Cranial nerves"]


def resample_streamline(s, step):
    d = np.linalg.norm(np.diff(s, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    L = cum[-1]
    n = max(2, int(np.ceil(L / step)) + 1)
    t = np.linspace(0, L, n)
    return np.stack([np.interp(t, cum, s[:, k]) for k in range(3)], axis=1), L


def simplify_polyline(p, eps=0.15, max_seg=5.0):
    """Douglas-Peucker on a densely sampled polyline, also capping segment length."""
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = p[b] - p[a]
        L = np.linalg.norm(seg)
        rel = p[a + 1:b] - p[a]
        if L < 1e-9:
            d = np.linalg.norm(rel, axis=1)
        else:
            d = np.linalg.norm(np.cross(rel, seg / L), axis=1)
        i = int(np.argmax(d))
        if d[i] > eps or L > max_seg:
            k = a + 1 + i if d[i] > eps else (a + b) // 2
            keep[k] = True
            stack += [(a, k), (k, b)]
    return p[keep]


def build_tracts(man):
    import colorsys
    rng = np.random.default_rng(7)
    folder = os.path.join(RAW, "hcp842", "Atlas_80_Bundles", "bundles")
    files = sorted(f for f in os.listdir(folder) if f.endswith(".trk") and f != "IF0F_R.trk")  # IF0F_R duplicates IFOF_R
    bundles, pts, counts = [], [], []
    for fn in files:
        stem = fn[:-4]
        m = re.match(r"^(.*?)(?:_(L|R))?$", stem)
        abbr, side = m.group(1), m.group(2) or ""
        if stem == "F_L_R":
            abbr, side = "F", ""
        full, cat = TRACTS[abbr]
        sl = nib.streamlines.load(os.path.join(folder, fn)).streamlines
        n = len(sl)
        keep = min(n, int(round(10 * np.sqrt(n))) + 15)
        order = rng.permutation(n)
        chosen = []
        for i in order:
            s = np.asarray(sl[i], np.float64)
            r, L = resample_streamline(s, 1.0)
            if L < 8:
                continue
            chosen.append(simplify_polyline(r))
            if len(chosen) >= keep:
                break
        first = len(counts)
        for r in chosen:
            pts.append(r)
            counts.append(len(r))
        bundles.append({"id": stem, "abbr": abbr, "name": full, "hemi": side, "category": cat,
                        "first": first, "count": len(chosen), "total": n})
    # colours: each category gets a hue band
    for ci, cat in enumerate(TRACT_CATS):
        bs = [b for b in bundles if b["category"] == cat]
        base = [0.58, 0.0, 0.33, 0.12, 0.8, 0.9][ci]
        for j, b in enumerate(bs):
            hue = (base + (j / max(1, len(bs))) * 0.16) % 1.0
            b["color"] = hexcol([c * 255 for c in colorsys.hls_to_rgb(hue, 0.6, 0.75)])
    P = np.concatenate(pts)
    log(f"tracts: {len(bundles)} bundles, {len(counts)} streamlines, {len(P)} points")
    q = np.round(P * 50).astype(np.int16)  # 0.02 mm steps
    man["tracts"] = {"label": "White-matter tracts",
                     "description": "HCP-842 population-averaged tractography atlas, 80 bundles (Yeh et al. 2018)",
                     "categories": TRACT_CATS, "positions": save("tracts/pos", q), "posScale": 0.02,
                     "counts": save("tracts/counts", np.array(counts, np.uint16)), "bundles": bundles}


# ----------------------------------------------------------------------------- volumes

def to_u8(a, lo, hi):
    return np.clip(np.round((a - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)


def build_volumes(man):
    vols = []
    ref = nib.load(TF_MNI + "res-01_T1w.nii.gz")
    shape = ref.shape
    aff = ref.affine
    assert np.allclose(aff[:3, :3], np.eye(3)), "expected 1 mm axis-aligned grid"

    def add_scalar(vid, label, desc, fname, kind="intensity", soft_mask=False):
        img = nib.load(TF_MNI + fname)
        a = np.asarray(img.dataobj, dtype=np.float32)
        assert img.shape == shape
        if soft_mask:
            # the skull-stripped template ends at a binary, voxel-aligned mask edge; soften it so
            # iso-surfaces there don't show staircases
            m = np.asarray(nib.load(TF_MNI + "res-01_desc-brain_mask.nii.gz").dataobj) > 0
            a = np.asarray(nib.load(TF_MNI + "res-01_T1w.nii.gz").dataobj, dtype=np.float32)
            a = a * ndimage.gaussian_filter(m.astype(np.float32), 1.2)
        if kind == "prob":
            u = to_u8(a, 0, 1)
            lo, hi = 0.0, 1.0
        else:
            nz = a[a > 0]
            lo, hi = 0.0, float(np.percentile(nz, 99.7))
            u = to_u8(a, lo, hi)
        vols.append({"id": vid, "label": label, "description": desc, "kind": kind,
                     "data": save(f"volumes/{vid}", u.ravel(order="F")), "min": lo, "max": hi})
        log(f"volume {vid}")

    add_scalar("t1brain", "T1 (brain extracted)", "MNI152 2009c T1-weighted template, skull-stripped",
               "res-01_desc-brain_T1w.nii.gz", soft_mask=True)
    add_scalar("t1", "T1 (full head)", "MNI152 2009c T1-weighted template", "res-01_T1w.nii.gz")
    add_scalar("t2", "T2 (full head)", "MNI152 2009c T2-weighted template", "res-01_T2w.nii.gz")
    add_scalar("gm", "Grey matter probability", "Tissue probability map", "res-01_label-GM_probseg.nii.gz", "prob")
    add_scalar("wm", "White matter probability", "Tissue probability map", "res-01_label-WM_probseg.nii.gz", "prob")
    add_scalar("csf", "CSF probability", "Tissue probability map", "res-01_label-CSF_probseg.nii.gz", "prob")

    # label volumes -> compact uint8 index + colour/name tables
    labels = []
    lut = read_fs_lut()
    a = np.asarray(nib.load(TF_MNI + "res-01_seg-aseg_dseg.nii.gz").dataobj).astype(np.int32)
    ids = [int(i) for i in np.unique(a) if i != 0]
    remap = np.zeros(256, np.uint8)
    names, colors = ["background"], ["#000000"]
    lower = lambda n: n[0].lower() + n[1:] if re.match(r"^[A-Z][a-z]", n) else n
    pretty = {251: "Corpus callosum (posterior)", 252: "Corpus callosum (mid-posterior)",
              253: "Corpus callosum (central)", 254: "Corpus callosum (mid-anterior)",
              255: "Corpus callosum (anterior)", 24: "CSF", 77: "White-matter hypointensities"}
    for labs, name, _, _ in ASEG:
        if len(labs) == 2:
            pretty.setdefault(labs[0], "Left " + lower(name))
            pretty.setdefault(labs[1], "Right " + lower(name))
        else:
            for l in labs:
                pretty.setdefault(l, name)
    for k, i in enumerate(ids, start=1):
        remap[i] = k
        n, c = lut.get(i, (str(i), [200, 200, 200]))
        if i in (2, 41):
            c = [236, 232, 220]
        names.append(pretty.get(i, n.replace("-", " ").replace("_", " ")))
        colors.append(hexcol(c))
    labels.append({"id": "aseg", "label": "Subcortical segmentation (aseg)",
                   "data": save("volumes/lab_aseg", remap[a].ravel(order="F")), "names": names, "colors": colors})

    mg = next(g for g in man.get("meshGroups", []) if g["id"] == "massp")
    massp = np.rint(np.asarray(nib.load(TF_MNI + "res-01_atlas-MASSP20_dseg.nii.gz").dataobj)).astype(np.int32)
    names, colors = ["background"] + [""] * int(massp.max()), ["#000000"] + ["#888888"] * int(massp.max())
    for s in mg["structures"]:
        k = int(s["id"].split("_")[1])
        names[k] = (s["hemi"] + " " if s["hemi"] else "") + s["name"]
        colors[k] = s["color"]
    labels.append({"id": "massp", "label": "Subcortical nuclei (MASSP)",
                   "data": save("volumes/lab_massp", massp.astype(np.uint8).ravel(order="F")),
                   "names": names, "colors": colors})

    tg = next(g for g in man.get("meshGroups", []) if g["id"] == "thalamic")
    thal = np.asarray(nib.load(TF_MNI + "res-01_atlas-MIAL67ThalamicNuclei_dseg.nii.gz").dataobj).astype(np.int32)
    names, colors = ["background"] + [""] * int(thal.max()), ["#000000"] + ["#888888"] * int(thal.max())
    for s in tg["structures"]:
        k = int(s["id"].split("_")[1])
        names[k] = s["hemi"] + " " + s["name"]
        colors[k] = s["color"]
    labels.append({"id": "thalamic", "label": "Thalamic nuclei (MIAL)",
                   "data": save("volumes/lab_thalamic", thal.astype(np.uint8).ravel(order="F")),
                   "names": names, "colors": colors})

    gm = np.asarray(nib.load(TF_MNI + "res-01_label-GM_probseg.nii.gz").dataobj)
    wm = np.asarray(nib.load(TF_MNI + "res-01_label-WM_probseg.nii.gz").dataobj)
    csf = np.asarray(nib.load(TF_MNI + "res-01_label-CSF_probseg.nii.gz").dataobj)
    st = np.stack([np.zeros_like(gm) + 0.2, csf, gm, wm])
    tissue = np.argmax(st, axis=0).astype(np.uint8)
    labels.append({"id": "tissue", "label": "Tissue class (GM / WM / CSF)",
                   "data": save("volumes/lab_tissue", tissue.ravel(order="F")),
                   "names": ["background", "CSF", "Grey matter", "White matter"],
                   "colors": ["#000000", "#3b7dd8", "#d0574f", "#ece8dc"]})

    man["volumes"] = {"dims": list(shape), "origin": aff[:3, 3].tolist(), "spacing": [1.0, 1.0, 1.0],
                      "scalars": vols, "labels": labels}


# ----------------------------------------------------------------------------- checks

# ----------------------------------------------------------------------------- TMS targets

TMS_GROUPS = {"Setting the dose": "#e9edf2", "Depression": "#ffb340", "OCD": "#5cc8ff",
              "Smoking cessation": "#7ddc6f", "Migraine": "#d39cff"}

# Published targets. "mni"/"tal" points are snapped to the nearest cortical vertex; "region" targets are the
# centre of an atlas region on each listed hemisphere (deep-TMS coils reach a broad area around it).
TMS_TARGETS = [
    {"id": "m1_hand", "name": "Motor hotspot (hand)", "group": "Setting the dose", "tal": (-37, -21, 58),
     "use": "Single pulses here make the thumb or index finger twitch. The weakest pulse that does it (the motor "
            "threshold) sets the strength of every treatment session.",
     "how": "Found by moving the coil over the scalp while watching the hand.",
     "coil": "Figure-8 coil",
     "source": "Peak of primary motor cortex in a meta-analysis of motor imaging studies (Mayka et al. 2006), "
               "Talairach −37, −21, 58 converted to MNI"},
    {"id": "dlpfc_5cm", "name": "Left DLPFC: 5-cm rule", "group": "Depression", "mni": (-41, 16, 54),
     "use": "The target of the trials behind the first FDA clearance of TMS for depression (2008).",
     "how": "5 cm forward along the scalp from the motor hotspot. In many heads this lands farther back than "
            "intended, on premotor cortex.",
     "coil": "Figure-8 coil",
     "source": "Average of published 5-cm sites (Fox et al. 2012)"},
    {"id": "dlpfc_f3", "name": "Left DLPFC: F3", "group": "Depression", "mni": (-37, 26, 49),
     "use": "A common depression target where MRI guidance isn't used.",
     "how": "F3 is a position of the EEG 10–20 system, found with tape-measure head measurements (the Beam F3 "
            "method, Beam et al. 2009).",
     "coil": "Figure-8 coil",
     "source": "Cortex under F3 (Herwig et al. 2003, as tabulated by Fox et al. 2012)"},
    {"id": "dlpfc_conn", "name": "Left DLPFC: connectivity target", "group": "Depression", "mni": (-38, 44, 26),
     "use": "Published left-DLPFC sites worked better the more their activity runs opposite to the subgenual "
            "cingulate's. This is the group-average best site.",
     "how": "Reached with MRI neuronavigation. The accelerated SAINT/SNT protocol finds each patient's own version "
            "from their fMRI (Cole et al. 2020, 2022).",
     "coil": "Figure-8 coil",
     "source": "Fox et al. 2012"},
    {"id": "dlpfc_right", "name": "Right DLPFC: F4", "group": "Depression", "mni": (37, 26, 49),
     "use": "Slow (1 Hz), calming stimulation here is an alternative or an addition to left-sided treatment.",
     "how": "F4 is the right-hand twin of F3 in the 10–20 system. Shown here as the mirror image of the F3 site.",
     "coil": "Figure-8 coil",
     "source": "Mirror of the F3 site (Herwig et al. 2003)"},
    {"id": "ocd", "name": "Medial prefrontal & anterior cingulate", "group": "OCD",
     "region": ("dk", "caudalanteriorcingulate", "LR"), "scalp": "ocd",
     "use": "Deep TMS with the H7 coil, FDA-cleared as an add-on treatment for OCD (2018).",
     "how": "The coil sits on the midline, 4 cm in front of the spot that makes the leg twitch.",
     "coil": "H7 deep-TMS coil",
     "source": "Carmi et al. 2018, 2019"},
    {"id": "smoking", "name": "Insula & lateral prefrontal cortex", "group": "Smoking cessation",
     "region": ("dk", "insula", "LR"),
     "use": "Deep TMS with the H4 coil, FDA-cleared in 2020 as a short-term aid to quitting smoking. It reaches "
            "the lateral prefrontal cortex and the insula on both sides.",
     "how": "The coil is a helmet; its position is set from the motor hotspot.",
     "coil": "H4 deep-TMS coil",
     "source": "Zangen et al. 2021"},
    {"id": "migraine", "name": "Occipital cortex", "group": "Migraine",
     "region": ("destrieux", "Pole_occipital", "LR"),
     "use": "A handheld single-pulse device, FDA-cleared for migraine with aura (2013). The pulse is thought to "
            "interrupt the wave of cortical activity behind the aura.",
     "how": "Held against the back of the head by the patient at the start of an attack.",
     "coil": "Handheld single-pulse device",
     "source": "Lipton et al. 2010"},
]


def tal_to_mni(p):
    """Talairach -> MNI by inverting Brett's mni2tal (above the AC), the conversion Mayka et al. 2006 used."""
    y, z = np.linalg.solve([[0.9688, 0.0460], [-0.0485, 0.9189]], [p[1], p[2]])
    return np.array([p[0] / 0.99, y, z])


def load_saved(desc):
    """Read back an array written by save()/save_quantized()."""
    with open(os.path.join(ROOT, "web", desc["url"]), "rb") as fh:
        a = np.frombuffer(gzip.decompress(fh.read()), dtype=desc["dtype"])
    return np.asarray(desc["min"]) + a.reshape(-1, 3) * np.asarray(desc["scale"]) if "scale" in desc else a


def build_tms(man):
    """Place the published TMS targets on the cortex, and each coil position on the scalp (the head surface)."""
    pial, atl = {}, {}
    for h in "LR":
        pial[h] = apply_affine(MNI305_TO_152, nib.load(f"{TF_FS}hemi-{h}_den-164k_pial.surf.gii").agg_data()[0])
        for aid, key in (("dk", "atlas-Desikan2006_seg-aparc_desc-curated"), ("destrieux", "atlas-Destrieux2009"),
                         ("brainnetome", "atlas-brainnetome")):
            atl[aid, h] = read_atlas(h, aid, key)[:2]

    g = next(g for g in man["meshGroups"] if g["id"] == "context")
    it = next(s for s in g["structures"] if s["id"] == "hull_head")
    hv = load_saved(g["positions"])[it["vStart"]:it["vStart"] + it["vCount"]]
    hf = load_saved(g["indices"]).reshape(-1, 3)[it["iStart"] // 3:(it["iStart"] + it["iCount"]) // 3] - it["vStart"]
    hn = np.zeros_like(hv)
    fn = np.cross(hv[hf[:, 1]] - hv[hf[:, 0]], hv[hf[:, 2]] - hv[hf[:, 0]])
    for k in range(3):
        np.add.at(hn, hf[:, k], fn)
    hn /= np.linalg.norm(hn, axis=1, keepdims=True)

    def scalp(p):
        i = int(np.argmin(((hv - p) ** 2).sum(1)))
        return hv[i], hn[i]

    def region_centre(aid, name, h):
        lab, names = atl[aid, h]
        return pial[h][lab == [n.lower() for n in names].index(name.lower())].mean(0)

    rnd = lambda a: [round(float(x), 2) for x in a]
    targets = []
    for t in TMS_TARGETS:
        points = []
        if "region" in t:
            aid, name, hemis = t["region"]
            centres = [region_centre(aid, name, h) for h in hemis]
            if t["id"] in ("ocd", "migraine"):  # one midline target for both hemispheres
                centres = [np.mean(centres, 0) * [0, 1, 1]]
            mni = None
        else:
            mni = tal_to_mni(t["tal"]) if "tal" in t else np.array(t["mni"], float)
            h = "L" if mni[0] < 0 else "R"
            centres = [pial[h][np.argmin(((pial[h] - mni) ** 2).sum(1))]]
        for c in centres:
            if t.get("scalp") == "ocd":
                # the leg spot (scalp over Brainnetome's leg motor area, both hemispheres), then 4 cm forward
                # along the midline
                leg, _ = scalp(np.mean([region_centre("brainnetome", "a4ll", h) for h in "LR"], 0) * [0, 1, 1])
                mid = np.nonzero((np.abs(hv[:, 0]) < 3) & (hv[:, 1] > leg[1]))[0]
                i = mid[np.argmin(np.abs(np.linalg.norm(hv[mid] - leg, axis=1) - 40))]
                s, n = hv[i], hn[i]
            else:
                s, n = scalp(c)
            points.append({"cortex": rnd(c), "scalp": rnd(s), "normal": rnd(n)})
        targets.append({k: t[k] for k in ("id", "name", "group", "use", "how", "coil", "source")} |
                       {"color": TMS_GROUPS[t["group"]], "broad": "region" in t,
                        "mni": rnd(mni) if mni is not None else None, "points": points})
        log(f"tms {t['id']}: " + "; ".join(f"cortex {p['cortex']} scalp {p['scalp']}" for p in points))
    man["tms"] = {"description": "Where common TMS treatments aim, from published studies. Positions are group "
                                 "averages on the template brain. Clinics find each patient's own target.",
                  "groups": list(TMS_GROUPS), "targets": targets}


# ----------------------------------------------------------------------------- signal flow

# Diffusion tractography can't tell which way a fibre conducts. The direction of each bundle comes from
# classical anatomy instead: per bundle, which end sends, what sits at each end, and a note for the viewer.
#   way: "two" (reciprocal), "one" (source end -> other end), "mid" (from a point midway out to both ends)
#   src: how to tell the source end apart: high/low (z), front/back (y), lateral/medial (|x|),
#        near:<mask> / far:<mask> (distance to a subcortical structure)
#   nodes: (source end, other end); "dk" = the Desikan-Killiany area at that end ("dk@lateral": at the
#        streamline's most lateral point instead), "thal" = the thalamic relay for the cortical area at the
#        other end, "<key>|side" = a deep node per hemisphere
FLOW_NODES = {
    "eye": "Eye (retina)", "lgn": "Thalamus: visual relay (LGN)", "mgn": "Thalamus: hearing relay (MGN)",
    "vpl": "Thalamus: body-sense relay (VPL/VPM)", "vl": "Thalamus: motor relay (VA/VL)",
    "md": "Thalamus: mediodorsal nucleus", "ant": "Thalamus: anterior nucleus", "pul": "Thalamus: pulvinar",
    "striatum": "Striatum (caudate & putamen)", "pons": "Pontine nuclei", "cord_motor": "Spinal cord (to the muscles)",
    "cord_sense": "Spinal cord (pain & temperature)", "dcn": "Dorsal column nuclei (touch & limb position)",
    "cbx": "Cerebellar cortex", "dcbn": "Deep cerebellar nuclei", "olive": "Inferior olive",
    "red": "Red nucleus", "ic": "Inferior colliculus", "cochlear": "Cochlear nuclei", "ear": "Inner ear",
    "vestib": "Vestibular nuclei", "eyemotor": "Eye-movement nuclei (midbrain)", "eyemuscles": "Eye muscles",
    "face_sense": "Face & mouth (touch)", "trigeminal": "Trigeminal nuclei", "facial_nuc": "Facial nucleus",
    "face_muscles": "Face muscles", "reticular": "Reticular formation", "hypothalamus": "Hypothalamus",
    "medulla_auto": "Medulla (autonomic centres)", "hippocampus": "Hippocampus",
    "mammillary": "Mammillary bodies", "pretectum": "Pretectum (midbrain)",
}
ASSOC_NOTE = "Two-way: association fibres carry signals in both directions between these areas."
COMM_NOTE = "Two-way: links matching areas of the two hemispheres, in both directions."
FLOW = {
    **{a: ("two", None, ("dk", "dk"), ASSOC_NOTE)
       for a in ["AF", "AST", "C", "EMC", "IFOF", "ILF", "MdLF", "SLF", "UF", "VOF"]},
    **{a: ("two", None, ("dk", "dk"), COMM_NOTE) for a in ["CC", "CCMid", "CC_ForcepsMajor", "CC_ForcepsMinor", "AC"]},
    "PC": ("two", None, ("pretectum|side", "pretectum|side"), "Two-way: joins the pretectal areas (pupil reflex)."),
    "OR": ("one", "near:thalamus", ("lgn|side", "dk"), "One-way: from the thalamus (LGN) up to the visual cortex."),
    # the atlas's acoustic radiation runs on past Heschl's gyrus into parietal white matter; its most lateral
    # point is where it passes the auditory cortex
    "AR": ("one", "near:thalamus", ("mgn|side", "dk@lateral"),
           "One-way: from the thalamus (MGN) up to the auditory cortex."),
    "CT": ("two", "near:thalamus", ("thal", "dk"),
           "Two-way: thalamus to cortex and back; each cortical area has its own thalamic relay."),
    "CS": ("one", "far:striatum", ("dk", "striatum|side"), "One-way: from the cortex down to the striatum."),
    "CST": ("one", "high", ("dk", "cord_motor"), "One-way: from motor cortex down to the spinal cord (movement)."),
    **{a: ("one", "high", ("dk", "pons"), "One-way: from the cortex down to the pons, then on to the cerebellum.")
       for a in ["FPT", "OPT", "PPT", "TPT"]},
    "F": ("one", "back", ("hippocampus|side", "mammillary"), "One-way: from the hippocampus to the mammillary bodies."),
    "CB": ("two", None, ("cbx|side", "cbx|side"), "Two-way: fibres within the cerebellum."),
    "V": ("two", None, ("cbx|side", "cbx|side"), "Two-way: across the vermis, between the cerebellar hemispheres."),
    "ICP": ("one", "low", ("olive|side", "cbx|side"),
            "One-way: from the medulla (inferior olive) into the cerebellum."),
    "MCP": ("mid", None, ("cbx|side", "cbx|side"), "One-way: from the pons outward into both cerebellar hemispheres."),
    "SCP": ("one", "back", ("dcbn|side", "red|side"),
            "One-way: the cerebellum's output, to the red nucleus and motor thalamus."),
    "CTT": ("one", "high", ("red|side", "olive|side"), "One-way: from the red nucleus down to the inferior olive."),
    "DLF": ("two", "high", ("hypothalamus", "medulla_auto"), "Two-way: hypothalamus and brainstem autonomic centres."),
    "LL": ("one", "low", ("cochlear|side", "ic|side"),
           "One-way: hearing, up from the cochlear nuclei to the midbrain."),
    "ML": ("one", "low", ("dcn|side", "vpl|side"), "One-way: touch and limb position, up to the thalamus."),
    "MLF": ("two", "low", ("vestib|side", "eyemotor|side"), "Two-way: links balance and eye-movement nuclei."),
    "RST": ("one", "high", ("reticular|side", "cord_motor"), "One-way: from the brainstem down to the spinal cord."),
    "STT": ("one", "low", ("cord_sense|side", "vpl|side"), "One-way: pain and temperature, up to the thalamus."),
    "CNII": ("one", "front", ("eye|side", "lgn|side"), "One-way: from the eye back to the thalamus (LGN)."),
    "CNIII": ("one", "back", ("eyemotor|side", "eyemuscles|side"),
              "One-way: from the midbrain out to the eye muscles."),
    "CNIV": ("one", "back", ("eyemotor|side", "eyemuscles|side"), "One-way: from the midbrain out to an eye muscle."),
    "CNV": ("one", "lateral", ("face_sense|side", "trigeminal|side"),
            "Mostly one-way: touch from the face and mouth into the brainstem."),
    "CNVII": ("one", "medial", ("facial_nuc|side", "face_muscles|side"),
              "One-way: from the brainstem out to the face muscles."),
    "CNVIII": ("one", "lateral", ("ear|side", "cochlear|side"), "One-way: from the inner ear into the brainstem."),
}
# Hops with no drawn bundle (short or unmapped pathways): (from, to, delay ms, what it is)
FLOW_RELAYS = [
    ("ic.{s}", "mgn.{s}", 2, "brachium of the inferior colliculus"),
    ("cochlear.{s}", "ic.{o}", 3, "crossed hearing pathway (superior olive)"),
    ("red.{s}", "vl.{s}", 2, "cerebellar output continuing to the motor thalamus"),
    ("cbx.{s}", "dcbn.{s}", 3, "Purkinje cells to the deep cerebellar nuclei"),
    ("trigeminal.{s}", "vpl.{o}", 3, "trigeminothalamic tract"),
    ("{s}.entorhinal", "hippocampus.{s}", 3, "perforant path"),
    ("mammillary", "ant.{s}", 3, "mammillothalamic tract"),
    ("striatum.{s}", "vl.{s}", 8, "basal ganglia loop through the pallidum"),
    ("vestib.{s}", "cbx.{s}", 3, "vestibulocerebellar fibres"),
]
THAL_RELAY = {
    "vl": ["precentral", "caudalmiddlefrontal", "paracentral"], "vpl": ["postcentral"],
    "mgn": ["transversetemporal"],
    "ant": ["rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate", "isthmuscingulate"],
    "md": ["superiorfrontal", "rostralmiddlefrontal", "parsopercularis", "parstriangularis", "parsorbitalis",
           "lateralorbitofrontal", "medialorbitofrontal", "frontalpole"],
}  # anything else: pulvinar


def build_flow(man):
    """Per streamline: the node at each end and the direction(s) signals run, from the FLOW rules."""
    from scipy.spatial import cKDTree
    T = man["tracts"]
    P = load_saved(T["positions"]).reshape(-1, 3) * T["posScale"]
    counts = load_saved(T["counts"]).astype(np.int64)
    start = np.concatenate([[0], np.cumsum(counts)])
    ends = np.stack([P[start[:-1]], P[start[1:] - 1]], 1)  # streamline, end, xyz

    # cortex: nearest labelled white-surface vertex
    cv, cl = [], []
    dk = {}
    for h in "LR":
        w = apply_affine(MNI305_TO_152, nib.load(f"{TF_FS}hemi-{h}_den-164k_white.surf.gii").agg_data()[0])
        lab, names, _ = read_atlas(h, "dk", "atlas-Desikan2006_seg-aparc_desc-curated")
        keep = lab != 0
        cv.append(w[keep])
        cl += [f"{h}.{names[i]}" for i in lab[keep]]
        dk[h] = names
    ctree, cl = cKDTree(np.concatenate(cv)), np.array(cl)
    img = nib.load(TF_MNI + "res-01_seg-aseg_dseg.nii.gz")
    aseg = np.asarray(img.dataobj)
    masks = {"thalamus": [10, 49], "striatum": [11, 50, 12, 51, 13, 52]}
    mtree = {k: cKDTree(apply_affine(img.affine, np.argwhere(np.isin(aseg, v)).astype(float)))
             for k, v in masks.items()}

    nodes, index = [], {}

    def node(nid):
        if nid not in index:
            key, _, side = nid.partition(".")
            if key in ("L", "R"):  # cortical area
                name = ("Left " if key == "L" else "Right ") + pretty_dk(side).lower()
                nodes.append({"id": nid, "name": name, "kind": "cortex", "hemi": key, "dk": side})
            else:
                name = FLOW_NODES[key]
                if side:
                    name = ("Left " if side == "L" else "Right ") + name[0].lower() + name[1:]
                nodes.append({"id": nid, "name": name, "kind": "deep", "hemi": side})
            index[nid] = len(nodes) - 1
        return index[nid]

    side = lambda p: "L" if p[0] < 0 else "R"
    relay_of = {a: k for k, v in THAL_RELAY.items() for a in v}
    links = np.zeros((len(counts), 4), np.uint16)
    notes = {}
    for b in T["bundles"]:
        way, src, (na, nb), note = FLOW[b["abbr"]]
        notes[b["id"]] = note
        for s in range(b["first"], b["first"] + b["count"]):
            e = ends[s]
            if src is None:
                first = 0
            else:
                kind, _, arg = src.partition(":")
                score = {"high": e[:, 2], "low": -e[:, 2], "front": e[:, 1], "back": -e[:, 1],
                         "lateral": np.abs(e[:, 0]), "medial": -np.abs(e[:, 0])}.get(kind)
                if kind in ("near", "far"):
                    d = mtree[arg].query(e)[0]
                    score = -d if kind == "near" else d
                first = int(np.argmax(score))  # the source end
            pts = (e[first], e[1 - first])
            cort = [cl[ctree.query(p)[1]] for p in pts]
            ids = []
            for k, (spec, p) in enumerate(zip((na, nb), pts)):
                if spec == "dk":
                    ids.append(cort[k])
                elif spec == "dk@lateral":
                    q = P[start[s]:start[s + 1]]
                    ids.append(cl[ctree.query(q[np.argmax(np.abs(q[:, 0]))])[1]])
                elif spec == "thal":
                    ids.append(f"{relay_of.get(cort[1 - k].split('.')[1], 'pul')}.{side(p)}")
                else:
                    key, _, per = spec.partition("|")
                    ids.append(f"{key}.{side(p)}" if per else key)
            a, bb = (ids if first == 0 else ids[::-1])
            mode, mid, m = (3 if way == "two" else 1 if first == 0 else 2), 0, 65535
            if way == "mid":
                # MCP: signals start where the streamline is nearest the pons (the midline) and run outward;
                # a streamline that ends at the midline runs one way, from that end
                q = P[start[s]:start[s + 1]]
                i = int(np.argmin(np.abs(q[:, 0])))
                f = np.linalg.norm(np.diff(q[:i + 1], axis=0), axis=1).sum() / \
                    max(1e-6, np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
                if f < 0.08:
                    a, mode = "pons", 1
                elif f > 0.92:
                    bb, mode = "pons", 2
                else:
                    m, mode, mid = node("pons"), 4, int(round(f * 255))
            links[s] = [node(a), node(bb), m, mode | (mid << 8)]
    relays = []
    for fr, to, delay, what in FLOW_RELAYS:
        for s_, o in (("L", "R"), ("R", "L")):
            f_, t_ = fr.format(s=s_, o=o), to.format(s=s_, o=o)
            if f_ in index or f_.startswith(("L.", "R.")):
                r = [node(f_), node(t_), delay, what]
                if r not in relays:
                    relays.append(r)
    log(f"flow: {len(nodes)} nodes, {len(relays)} relays, "
        f"{np.bincount(links[:, 3] & 15, minlength=5).tolist()} streamlines by mode (-, A>B, B>A, both, mid)")
    man["flow"] = {
        "description": "An illustrative model, not a recording: anatomy from the tract atlas, the direction of "
                       "each pathway from established anatomy, and timing modelled.",
        "nodes": nodes, "links": save("flow/links", links), "relays": relays, "notes": notes}


# ----------------------------------------------------------------------------- F-TRACT

# The F-TRACT atlas (v2112, adults): responses to single-pulse electrical stimulation recorded with intracranial
# electrodes in 780 epilepsy patients, between the Lausanne2008-33 parcels (the Desikan-Killiany areas plus aseg
# structures). Free for research use only, so it lives in the private `dynamic` branch; tools/fetch_ftract.py
# gets the files.
FTRACT = os.path.join(RAW, "ftract", "extracted", "ages_15_100", "sr_8.40", "seg_None_None", "pl_200",
                      "Lausanne2008-33")
FTRACT_FEATURES = ["probability", "onset_latency", "peak_latency", "axonal_conduction_delay", "amplitude"]
FTRACT_DEEP = {"Hippocampus": ["hippocampus"], "Caudate": ["striatum"], "Putamen": ["striatum"],
               "Thalamus-Proper": ["lgn", "mgn", "vpl", "vl", "md", "ant", "pul"]}


def read_ftract_counts():
    """Number of recordings behind each probability, from the atlas pickle. Only numpy's array constructors may
    be unpickled, so the file can't run code."""
    import io
    import pickle
    try:
        from numpy._core import multiarray as ma, numeric as nu
    except ImportError:
        from numpy.core import multiarray as ma, numeric as nu
    allowed = {("numpy.core.multiarray", "_reconstruct"): ma._reconstruct, ("numpy", "ndarray"): np.ndarray,
               ("numpy", "dtype"): np.dtype, ("numpy.core.numeric", "_frombuffer"): nu._frombuffer}

    class Safe(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in allowed:
                raise pickle.UnpicklingError(f"blocked {module}.{name}")
            return allowed[module, name]

    path = os.path.join(FTRACT, "atlas__ages_15_100__sr_8.40__seg_None_None__pl_200__Lausanne2008-33.pkl.gz")
    feats = Safe(io.BytesIO(gzip.decompress(open(path, "rb").read()))).load()
    return np.asarray(feats[0]["count"])  # feature 0 is the probability (its mean over recordings)


def build_ftract(man):
    """Measured stimulation responses between areas, keyed to the flow nodes (run after flow)."""
    M, names = {}, None
    for f in FTRACT_FEATURES:
        with open(os.path.join(FTRACT, "export", f, f"{f}.csv")) as fh:
            rows = [r.rstrip("\n").split(",") for r in fh if not r.startswith("#")]
        assert names is None or rows[0][1:] == names
        names = rows[0][1:]
        assert [r[0] for r in rows[1:]] == names  # stimulated parcels down, recorded parcels across
        M[f] = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    count = read_ftract_counts()
    assert count.shape == M["probability"].shape

    index = {n["id"]: i for i, n in enumerate(man["flow"]["nodes"])}
    parcels = []
    for nm in names:
        m = re.match(r"^ctx-(lh|rh)-(\w+)$", nm)
        s = re.match(r"^(Left|Right)-(.+)$", nm)
        if m:
            h = "L" if m.group(1) == "lh" else "R"
            ids, label = [f"{h}.{m.group(2)}"], ("Left " if h == "L" else "Right ") + pretty_dk(m.group(2)).lower()
        elif s:
            h, what = s.group(1)[0], s.group(2)
            ids = [f"{k}.{h}" for k in FTRACT_DEEP.get(what, [])]
            label = f"{s.group(1)} " + {"Thalamus-Proper": "thalamus", "Accumbens-area": "nucleus accumbens"}.get(
                what, what.lower())
        else:
            ids, label = [], {"Brain-Stem": "Brainstem"}.get(nm, nm)
        parcels.append({"name": nm, "label": label, "nodes": [index[i] for i in ids if i in index]})

    # a response needs a probability and an onset latency; the axonal delay can be missing (NaN)
    P, on = M["probability"], M["onset_latency"]
    ii, jj = np.nonzero(~np.isnan(P) & ~np.isnan(on) & ~np.eye(len(names), dtype=bool))
    cols = ["stim", "rec", "probability", "onset", "peak", "delay", "amplitude", "recordings"]
    edges = np.stack([ii, jj, P[ii, jj], on[ii, jj], M["peak_latency"][ii, jj], M["axonal_conduction_delay"][ii, jj],
                      M["amplitude"][ii, jj], count[ii, jj]], 1).astype(np.float32)
    log(f"ftract: {len(names)} parcels, {len(edges)} measured responses "
        f"(median probability {np.median(edges[:, 2]):.2f}, onset {np.median(edges[:, 3]):.1f} ms, "
        f"{int(np.isnan(edges[:, 5]).sum())} without an axonal delay)")
    man["ftract"] = {
        "description": "Measured responses to single-pulse electrical stimulation, recorded with intracranial "
                       "electrodes in 780 epilepsy patients (F-TRACT atlas v2112, ages 15 and up).",
        "source": "F-TRACT atlas (Trebaul et al. 2018; Lemaréchal et al. 2022), free for research use only",
        "parcels": parcels, "columns": cols, "edges": save("ftract/edges", edges)}


def check_alignment():
    """Mean GM probability sampled at fsaverage mid-thickness vertices, with/without MNI305->152."""
    img = nib.load(TF_MNI + "res-01_label-GM_probseg.nii.gz")
    gm = np.asarray(img.dataobj)
    inv = np.linalg.inv(img.affine)
    for h in "LR":
        v, _ = nib.load(f"{TF_FS}hemi-{h}_den-164k_midthickness.surf.gii").agg_data()
        for name, m in [("identity", np.eye(4)), ("305->152", MNI305_TO_152)]:
            vox = apply_affine(inv @ m, v.astype(np.float64))
            s = ndimage.map_coordinates(gm, vox.T, order=1)
            print(f"  {h} {name:9s} mean GM prob = {s.mean():.4f}")
        best = None
        for dx in (-2, 0, 2):
            for dy in (-2, 0, 2):
                for dz in (-2, 0, 2):
                    t = MNI305_TO_152.copy()
                    t[:3, 3] += [dx, dy, dz]
                    s = ndimage.map_coordinates(gm, apply_affine(inv @ t, v.astype(np.float64)).T, order=1).mean()
                    if best is None or s > best[0]:
                        best = (s, dx, dy, dz)
        print(f"  {h} best shift around 305->152: {best}")


def main():
    steps = sys.argv[1:] or ["cortex", "meshes", "tracts", "flow", "ftract", "volumes", "tms"]
    os.makedirs(OUT, exist_ok=True)
    if "check" in steps:
        check_alignment()
        steps = [s for s in steps if s != "check"]
    man = load_manifest()
    man["frame"] = "MNI152NLin2009cAsym (RAS, mm)"
    for s in steps:
        {"cortex": build_cortex, "atlases": build_atlases, "meshes": build_meshes, "tracts": build_tracts,
         "flow": build_flow, "ftract": build_ftract, "volumes": build_volumes, "tms": build_tms}[s](man)
        with open(MANIFEST_PATH, "w") as fh:
            json.dump(man, fh, indent=1)
    log("done")


if __name__ == "__main__":
    main()
