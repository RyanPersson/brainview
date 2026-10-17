# Data credits and licenses

Everything in `web/data/` is derived from the open datasets below, and each one stays under its
own terms. The MIT License in [LICENSE](LICENSE) covers only Brainview's code, not this data.
None of the files are the originals. Brainview has converted all of them to compact
binaries, moved the surfaces from MNI305 into MNI152 space, quantized positions to 16 bits and
intensities to 8 bits, and turned label maps into colour-table indices. It has also computed the
inflated surfaces, the pial–white distance map and every 3D mesh from the volumes, and subsampled
and simplified the streamlines.

The license texts that have to travel with the data are in [`web/licenses/`](web/licenses/), so
they're deployed with the site as well.

## Human Connectome Project

Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium (Principal
Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and
Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for
Systems Neuroscience at Washington University.

Four layers are derived from WU-Minn HCP Open Access data: the Glasser HCP-MMP1 parcellation, the
HCP-842 tracts, the MIAL thalamic nuclei and the Brainnetome parcellation. Brainview redistributes
them under the same
[WU-Minn HCP Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)
([copy](web/licenses/HCP-Open-Access-Data-Use-Terms.txt)). Anyone who reuses these files must:

- not try to identify or contact the people who were scanned;
- pass the data on only under the same terms;
- include the acknowledgment above in public presentations.

## Datasets

| Data | Used for | License / terms | Cite |
|---|---|---|---|
| FreeSurfer *fsaverage*: pial, white and sphere surfaces, sulc, curv | Cortex shapes, depth and curvature colours | [FreeSurfer license](web/licenses/FreeSurfer.txt) | Fischl et al. 1999, [doi:10.1002/(SICI)1097-0193(1999)8:4<272::AID-HBM10>3.0.CO;2-4](https://doi.org/10.1002/(SICI)1097-0193(1999)8:4%3C272::AID-HBM10%3E3.0.CO;2-4) |
| Desikan–Killiany atlas (FreeSurfer 7) | Cortex colours | [FreeSurfer license](web/licenses/FreeSurfer.txt) | Desikan et al. 2006, [doi:10.1016/j.neuroimage.2006.01.021](https://doi.org/10.1016/j.neuroimage.2006.01.021) |
| Destrieux atlas (FreeSurfer 7) | Cortex colours | [FreeSurfer license](web/licenses/FreeSurfer.txt) | Destrieux et al. 2010, [doi:10.1016/j.neuroimage.2010.06.010](https://doi.org/10.1016/j.neuroimage.2010.06.010) |
| Yeo 7/17 networks (FreeSurfer 7) | Cortex colours | [FreeSurfer license](web/licenses/FreeSurfer.txt) | Yeo et al. 2011, [doi:10.1152/jn.00338.2011](https://doi.org/10.1152/jn.00338.2011) |
| Schaefer 2018, 100/400/1000 parcels | Cortex colours | MIT ([CBIG](https://github.com/ThomasYeoLab/CBIG)) | Schaefer et al. 2018, [doi:10.1093/cercor/bhx179](https://doi.org/10.1093/cercor/bhx179) |
| Lausanne 2018, scales 1/3/5 (Alemán-Gómez, Cammoun, Hagmann) | Cortex colours | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Cammoun et al. 2012, [doi:10.1016/j.jneumeth.2011.09.031](https://doi.org/10.1016/j.jneumeth.2011.09.031) |
| Brainnetome cortical parcellation | Cortex colours, and the body map (its motor and touch subregions grouped by body part) | [Non-commercial, share-alike](web/licenses/Brainnetome.txt) + HCP terms | Fan et al. 2016, [doi:10.1093/cercor/bhw157](https://doi.org/10.1093/cercor/bhw157) |
| HCP-MMP1.0 projected on fsaverage, by Kathryn Mills | Cortex colours (Glasser) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) + HCP terms | Glasser et al. 2016, [doi:10.1038/nature18933](https://doi.org/10.1038/nature18933); Mills 2016, [doi:10.6084/m9.figshare.3498446.v2](https://doi.org/10.6084/m9.figshare.3498446.v2) |
| ICBM 2009c nonlinear asymmetric template: T1, T2, GM/WM/CSF probabilities, brain mask | MRI slices, volume rendering, brain hull, head (skin surface of the T1) | [MNI notice](web/licenses/MNI-ICBM152.txt) | Fonov et al. 2011, [doi:10.1016/j.neuroimage.2010.07.033](https://doi.org/10.1016/j.neuroimage.2010.07.033); Fonov et al. 2009, [doi:10.1016/S1053-8119(09)70884-5](https://doi.org/10.1016/S1053-8119(09)70884-5) |
| aseg segmentation of that template (made with FreeSurfer, distributed by TemplateFlow) | Subcortical structures, label overlay | [MNI notice](web/licenses/MNI-ICBM152.txt); label names and colours from FreeSurferColorLUT ([FreeSurfer license](web/licenses/FreeSurfer.txt)) | as for the template |
| MASSP 2.0 subcortical atlas (Bazin, Alkemade, Forstmann, Miletic) | Subcortical nuclei | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Bazin et al. 2025, [doi:10.1162/imag_a_00560](https://doi.org/10.1162/imag_a_00560) |
| MIAL thalamic nuclei atlas | Thalamic nuclei | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) + HCP terms | Najdenovska et al. 2018, [doi:10.1038/sdata.2018.270](https://doi.org/10.1038/sdata.2018.270) |
| Advanced Atlas of 80 Bundles in MNI space, © Eleftherios Garyfallidis, curated from the HCP-842 atlas | White-matter tracts | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) + HCP terms | Yeh et al. 2018, [doi:10.1016/j.neuroimage.2018.05.027](https://doi.org/10.1016/j.neuroimage.2018.05.027); Garyfallidis et al. 2018, [doi:10.1016/j.neuroimage.2017.07.015](https://doi.org/10.1016/j.neuroimage.2017.07.015) |

Where the files came from:

- The fsaverage and MNI152NLin2009cAsym data came from [TemplateFlow](https://www.templateflow.org)
  (Ciric et al. 2022, [doi:10.1038/s41592-022-01681-2](https://doi.org/10.1038/s41592-022-01681-2)).
- The HCP-MMP1 annotation came from figshare.
- The 80 bundles came from figshare, through DIPY's fetcher. The bundles are in ICBM 2009a space
  and are shown in 2009c without resampling.

The FreeSurfer license notes that this data was made for research and is not meant for clinical
use. Brainview is the same: it's for learning and exploring, not for diagnosis.

## Signal flow

The signal-flow links (`web/data/flow/`) are derived from the 80-bundle atlas, the Desikan–Killiany
atlas and the aseg segmentation, and keep their terms. The direction of each bundle is textbook
neuroanatomy, not data. The deep-sleep scenario follows Massimini et al. 2004,
[doi:10.1523/JNEUROSCI.1318-04.2004](https://doi.org/10.1523/JNEUROSCI.1318-04.2004).

## TMS targets

The TMS layer places published coordinates and procedures on the template. They are facts from these
papers, not data files:

- The 5-cm rule and connectivity targets, and the cortex under F3: Fox et al. 2012,
  [doi:10.1016/j.biopsych.2012.04.028](https://doi.org/10.1016/j.biopsych.2012.04.028), which
  tabulates Herwig et al. 2003, [doi:10.1023/B:BRAT.0000006333.93597.9d](https://doi.org/10.1023/B:BRAT.0000006333.93597.9d).
- The Beam F3 method: Beam et al. 2009, [doi:10.1016/j.brs.2008.09.006](https://doi.org/10.1016/j.brs.2008.09.006).
- SAINT/SNT: Cole et al. 2020, [doi:10.1176/appi.ajp.2019.19070720](https://doi.org/10.1176/appi.ajp.2019.19070720),
  and 2022, [doi:10.1176/appi.ajp.2021.20101429](https://doi.org/10.1176/appi.ajp.2021.20101429).
- The motor hotspot: the M1 peak of Mayka et al. 2006, [doi:10.1016/j.neuroimage.2006.02.004](https://doi.org/10.1016/j.neuroimage.2006.02.004).
- OCD (H7 coil): Carmi et al. 2018, [doi:10.1016/j.brs.2017.09.004](https://doi.org/10.1016/j.brs.2017.09.004),
  and 2019, [doi:10.1176/appi.ajp.2019.18101180](https://doi.org/10.1176/appi.ajp.2019.18101180).
- Smoking cessation (H4 coil): Zangen et al. 2021, [doi:10.1002/wps.20905](https://doi.org/10.1002/wps.20905).
- Migraine (single-pulse TMS): Lipton et al. 2010, [doi:10.1016/S1474-4422(10)70054-5](https://doi.org/10.1016/S1474-4422(10)70054-5).

Some positions are approximations made here, not published coordinates:
- The right-DLPFC point is the mirror image of the F3 site.
- The OCD, smoking and migraine points are centres of atlas regions (Desikan–Killiany anterior
  cingulate and insula, Destrieux occipital pole).
- The OCD coil spot is placed 4 cm in front of the scalp above Brainnetome's leg motor area.

The body map's note on effector and inter-effector zones refers to Gordon et al. 2023,
[doi:10.1038/s41586-023-05964-2](https://doi.org/10.1038/s41586-023-05964-2).

## Libraries

[three.js](https://threejs.org) and [three-mesh-bvh](https://github.com/gkjohnson/three-mesh-bvh)
are vendored in `web/vendor/` with their MIT license files.
