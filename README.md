# Brainview

An interactive three.js model of the human brain built from open neuroimaging data. Every layer lives in
one common space, MNI152NLin2009cAsym (millimetres, RAS), so the layers can be stacked, cut and compared.

**Open:** on GitHub Pages once deployed (see *Deploying*), or locally with `python3 serve.py` → http://127.0.0.1:8051/

## Layers

| Layer | What it is | Ways to look at it |
|---|---|---|
| Cortex | FreeSurfer *fsaverage*, 163,842 vertices per hemisphere | Morph between the pial, white, inflated, very inflated and sphere surfaces. Colour by sulcal depth, curvature or pial–white distance, or by 12 parcellations: Desikan–Killiany, Destrieux, Glasser HCP-MMP1, Brainnetome, Yeo 7/17, Schaefer 100/400/1000, Lausanne 1/3/5. Solid or glass styles. |
| Subcortical structures | Surfaces from the template's aseg segmentation | Thalamus, basal ganglia, hippocampus, amygdala, brainstem, cerebellum, ventricles, corpus callosum, and the cortex and white matter as volumetric segmentations. |
| Subcortical nuclei | MASSP atlas | 63 small nuclei (STN, SN, red nucleus, GPi/GPe, VTA, PAG…), hippocampal subfields, commissures. |
| Thalamic nuclei | MIAL atlas | 7 nuclear groups per thalamus. |
| White-matter tracts | HCP-842 tractography atlas, 80 bundles | Tube-shaded streamlines coloured by direction or by bundle. Density, width and category filters. |
| MRI slices | T1, T2 and GM/WM/CSF probability maps | Axial, coronal and sagittal planes, with an optional label overlay (aseg, MASSP, thalamic, tissue class). |
| Volume rendering | The same volumes, ray-marched on the GPU | Iso-surface, translucent, or maximum-intensity projection, optionally coloured by labels. |
| Context | Brain hull from the template's brain mask; head (skin) from the full-head T1 | Glass or solid. |

The cut-away plane clips every 3D layer. Hovering anything names it: the region in the current
atlas, the structure, the tract, or the label under a slice. The MNI coordinate is shown too.

## Layout

```
build/build.py   raw/ → web/data/  (cortex | meshes | tracts | volumes | check)
serve.py         plain no-cache static server for web/ (the viewer inflates the *.bin.gz data itself)
web/             index.html, style.css, js/ (main, cortex, meshes, tracts, volume, ui, …), vendor/ (three r186, three-mesh-bvh),
                 licenses/ (data license texts, deployed with the site)
raw/             downloaded datasets (not needed at runtime)
tools/shot.mjs   headless Chrome screenshots / smoke tests: node tools/shot.mjs "name::<js using B = window.brainview>"
```

The server runs as `brainview.service` (systemd user unit, 127.0.0.1:8051) behind
`tailscale serve --https=8459`.

## Deploying to GitHub Pages

`web/` is the complete static site, prebuilt data included, so there's nothing to build.
`.github/workflows/pages.yml` uploads `web/` and publishes it on every push to `main`. You can
also run it by hand from the Actions tab.

One-time setup on GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The site then appears at `https://<user>.github.io/<repo>/`. All paths are relative, so the repo
name doesn't matter.

## Rebuilding the data

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy scipy nibabel nilearn mne scikit-image dipy pooch fast-simplification
systemd-run --user --scope -p MemoryMax=3G .venv/bin/python -I build/build.py      # everything; or name steps
```

Downloads go into `raw/`:
- TemplateFlow S3 (`tpl-fsaverage`, `tpl-MNI152NLin2009cAsym`)
- figshare 5528816/5528819 (HCP-MMP1 annotations)
- figshare 13638644 (`Atlas_80_Bundles.zip`)

The fsaverage surfaces are moved from MNI305 into MNI152 with FreeSurfer's published affine. Run
`build.py check` to confirm the fit: it samples the template's GM probability at the
mid-thickness vertices.

## Credits and data licenses

Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium (Principal
Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and
Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for
Systems Neuroscience at Washington University.

The data in `web/data/` comes from these sources:

- FreeSurfer *fsaverage* and the ICBM 2009c template, via [TemplateFlow](https://www.templateflow.org);
- cortical atlases by Desikan, Destrieux, Yeo, Schaefer, Fan (Brainnetome), Cammoun (Lausanne) and
  Glasser (HCP-MMP1, on fsaverage by Kathryn Mills);
- the MASSP 2.0 and MIAL thalamic atlases;
- the HCP-842 tract atlas (Yeh) as curated into 80 bundles by Garyfallidis.

Each dataset keeps its own license: FreeSurfer's license, the MNI notice, CC BY 4.0, MIT, the
Brainnetome atlas's non-commercial share-alike terms, and the
[WU-Minn HCP Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)
for the HCP-derived layers. [DATA-LICENSES.md](DATA-LICENSES.md) lists which terms apply to which
files, along with citations. The full license texts are in [web/licenses/](web/licenses/).

Brainview is for learning and exploring. It is not for clinical use.
