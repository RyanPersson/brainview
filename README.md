# Brainviewer

An interactive three.js model of the human brain built from open neuroimaging data. Every layer lives in
one common space, MNI152NLin2009cAsym (millimetres, RAS), so the layers can be stacked, cut and compared.

**Open:** on GitHub Pages once deployed (see *Deploying*), or locally with `python3 serve.py` → http://127.0.0.1:8051/

## Layers

| Layer | What it is | Ways to look at it |
|---|---|---|
| Cortex | FreeSurfer *fsaverage*, 163,842 vertices per hemisphere | Morph between the pial, white, inflated, very inflated and sphere surfaces. Colour by sulcal depth, curvature or pial–white distance, or by 12 parcellations: Desikan–Killiany, Destrieux, Glasser HCP-MMP1, Brainnetome, Yeo 7/17, Schaefer 100/400/1000, Lausanne 1/3/5. A body map shows which body part each stretch of the motor and touch strips serves (from Brainnetome), with labels on the surface. Solid or glass styles. |
| Subcortical structures | Surfaces from the template's aseg segmentation | Thalamus, basal ganglia, hippocampus, amygdala, brainstem, cerebellum, ventricles, corpus callosum, and the cortex and white matter as volumetric segmentations. |
| Subcortical nuclei | MASSP atlas | 63 small nuclei (STN, SN, red nucleus, GPi/GPe, VTA, PAG…), hippocampal subfields, commissures. |
| Thalamic nuclei | MIAL atlas | 7 nuclear groups per thalamus. |
| White-matter tracts | HCP-842 tractography atlas, 80 bundles | Tube-shaded streamlines coloured by direction or by bundle. Density, width and category filters. |
| MRI slices | T1, T2 and GM/WM/CSF probability maps | Axial, coronal and sagittal planes, with an optional label overlay (aseg, MASSP, thalamic, tissue class). |
| Volume rendering | The same volumes, ray-marched on the GPU | Iso-surface, translucent, or maximum-intensity projection, optionally coloured by labels. |
| Context | Brain hull from the template's brain mask; head (skin) from the full-head T1 | Glass or solid. |
| Signal flow | Signals travelling the tracts: an illustrative model, and measured stimulation responses from the F-TRACT atlas | Pulses flow along the streamlines as comets, bright head first, and the areas they reach light up. Modelled scenarios: resting activity, seeing, hearing, touch, movement, and a deep-sleep slow wave. Measured: stimulate one area (or each in turn) and watch the areas that responded light up at their recorded latencies. Gold or dark fibre-direction tract shading. Speed, brightness and glow width. Hovering a tract says which way it carries signals. |
| TMS targets | Published TMS treatment targets: depression (left DLPFC by the 5-cm rule, F3 and connectivity; right DLPFC), OCD, smoking cessation, migraine, and the motor hotspot that sets the dose | A dot on the brain target, a ring where the coil sits on the scalp, labels, and the use, method and source on hover. Group averages, for learning only. |

The cut-away plane clips every 3D layer. Hovering anything names it: the region in the current
atlas, the structure, the tract, or the label under a slice. The MNI coordinate is shown too.

## Signal flow

The Signal flow layer sends signals along the tracts. Its scenarios are either modelled from anatomy or
measured: stimulation responses from the F-TRACT atlas.

The modelled scenarios are a model, not a recording, and these are its parts:

- **Anatomy, from the data.** The streamlines are the tract atlas. `build.py flow` assigns each streamline
  end to the Desikan–Killiany area or the deep nucleus it reaches (thalamic relays, pons, cerebellum,
  brainstem nuclei, and the eye, ear, face or spinal cord at the edge of the atlas).
- **Direction, from anatomy.** Diffusion MRI can't tell which way a fibre conducts, so each bundle's
  direction comes from textbook neuroanatomy (the `FLOW` table in `build.py`). Projection, cerebellar,
  brainstem and cranial-nerve pathways run one way: optic radiation up from the LGN, corticospinal tract
  down to the cord, and so on. Association and commissural bundles carry signals both ways, so their
  direction at any moment depends on which end fired. A few short hops that the atlas has no bundle for,
  such as inferior colliculus to MGN, are relays without a drawn tract.
- **Timing, modelled.** One conduction speed, 3.9 m/s, the median that F-TRACT measured between
  cortical areas (see below), and 1.5 ms per relay. A node fires when enough input
  arrives, then rests for 60 ms. Each scenario sets how far a burst spreads. The deep-sleep scenario
  follows the front-to-back travel of slow waves (Massimini et al. 2004,
  [doi:10.1523/JNEUROSCI.1318-04.2004](https://doi.org/10.1523/JNEUROSCI.1318-04.2004)).

### Measured: stimulation responses (F-TRACT)

The F-TRACT atlas (v2112, [doi:10.5281/zenodo.7015415](https://doi.org/10.5281/zenodo.7015415), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); [David et al. 2013](https://doi.org/10.1016/j.neuroimage.2013.05.075),
[Trebaul et al. 2018](https://doi.org/10.1016/j.neuroimage.2018.07.039),
[Lemaréchal et al. 2022](https://doi.org/10.1093/brain/awab362)) pools cortico-cortical evoked potentials from 780 epilepsy
patients with intracranial electrodes: one area gets a single electrical pulse, and the others are
recorded. For each ordered pair of areas it gives how often a response appeared, its onset and peak
latency, and an estimated axonal conduction delay. That's a measured direction: A drives B.

`tools/fetch_ftract.py` pulls only the files used (the Lausanne2008-33 parcellation, adults 15 and up)
out of the 915 MB zip with HTTP range requests. `build.py ftract` maps the parcels onto the flow model's
nodes (68 cortical areas, plus hippocampus, caudate and putamen, and the thalamus as all its relay
nodes) and writes `web/data/ftract/`. Pallidum, accumbens, amygdala and brainstem have no node and are left out.

In the viewer the stimulated area flashes at time 0. Pulses run from it along the streamlines that join it
to each responding area, taking that pair's axonal delay, and each area lights at its onset latency,
which also counts time spent in synapses. Brightness follows the response probability; most measured
pairs respond more often than not (median 0.9), so the scale starts at 0.5. Pairs tested fewer times
than *Fewest recordings* (default 20) are left out: the thinly sampled ones run to extremes, such as
0 ms onsets. Responding areas with no streamline between them only light up. The fibres are the
tract atlas's, not the ones F-TRACT measured its distances on, so each pulse's speed is this
streamline's length over the measured delay.

The look follows Greg Dunn and Brian Edwards' *Self Reflected*: gold fibres on black, a sagittal cut,
and light running through the circuits.

## Layout

```
build/build.py   raw/ → web/data/  (cortex | atlases | meshes | tracts | flow | ftract | volumes | tms | check)
serve.py         plain no-cache static server for web/ (the viewer inflates the *.bin.gz data itself)
web/             index.html, style.css, js/ (main, cortex, meshes, tracts, flow, volume, ui, …), vendor/ (three r186, three-mesh-bvh),
                 licenses/ (data license texts, deployed with the site)
raw/             downloaded datasets (not needed at runtime)
tools/shot.mjs   headless Chrome screenshots / smoke tests: node tools/shot.mjs "name::<js using B = window.brainviewer>"
tools/fetch_ftract.py  the F-TRACT files the ftract step needs (range requests into the Zenodo zip)
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
- F-TRACT v2112, Zenodo 7015415: run `python3 -I tools/fetch_ftract.py` first (about 140 MB of transfer)

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
- the HCP-842 tract atlas (Yeh) as curated into 80 bundles by Garyfallidis;
- the F-TRACT atlas of responses to intracranial stimulation (INSERM and the F-TRACT consortium; David,
  Trebaul, Lemaréchal et al.).

Each dataset keeps its own license: FreeSurfer's license, the MNI notice, CC BY 4.0, MIT, the
Brainnetome atlas's non-commercial share-alike terms, and the
[WU-Minn HCP Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)
for the HCP-derived layers. [DATA-LICENSES.md](DATA-LICENSES.md) lists which terms apply to which
files, along with citations. The full license texts are in [web/licenses/](web/licenses/).
F-TRACT's authors also ask that their atlas be used for research only; Brainviewer is non-commercial.

Brainviewer is for learning and exploring. It is not for clinical use.

## License

Brainviewer's code is under the MIT License; see [LICENSE](LICENSE). That covers `web/index.html`,
`web/style.css`, `web/js/`, `build/`, `tools/`, `serve.py` and the workflow. The MIT License does
**not** cover:

- the data in `web/data/`, which keeps the licenses of its sources (see *Credits and data licenses*
  above and [DATA-LICENSES.md](DATA-LICENSES.md));
- the vendored libraries in `web/vendor/`, which come with their own MIT licenses.
