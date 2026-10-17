#!/usr/bin/env python3
"""Convert the downloaded open datasets (raw/) into compact binaries for the web viewer (web/data/).

Usage:  .venv/bin/python -I build/build.py [cortex] [meshes] [tracts] [volumes] [check]
        (no args = everything)

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
]


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

    # atlases
    for aid, label, key, desc in ATLASES:
        entry = {"id": aid, "label": label, "description": desc, "hemis": {}}
        for h in "LR":
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
    man["cortex"] = out


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
    steps = sys.argv[1:] or ["cortex", "meshes", "tracts", "volumes"]
    os.makedirs(OUT, exist_ok=True)
    if "check" in steps:
        check_alignment()
        steps = [s for s in steps if s != "check"]
    man = load_manifest()
    man["frame"] = "MNI152NLin2009cAsym (RAS, mm)"
    for s in steps:
        {"cortex": build_cortex, "meshes": build_meshes, "tracts": build_tracts, "volumes": build_volumes}[s](man)
        with open(MANIFEST_PATH, "w") as fh:
            json.dump(man, fh, indent=1)
    log("done")


if __name__ == "__main__":
    main()
