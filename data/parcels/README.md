# Network parcels: language, MD, ToM, physics

Group-level parcels for four networks, plus an alternative physics parcellation. Each subject's fROIs are defined within these parcels, typically as the top 10% of voxels by the localizer contrast inside each parcel. All files are in MNI space (SPM12 template grid): 91 × 109 × 91 voxels, 2 mm isotropic, the same affine for all. Each voxel holds an integer label (0 = outside every parcel).

| File | Network | Parcels | Localizer contrast | Source |
|---|---|---|---|---|
| `LANGUAGE_noAngG.nii.gz` | Language | 10 (5 per hemisphere) | Sentences > Nonwords (S-N) | derived from `LANGUAGE_all.nii.gz` (see below) |
| `MD.nii.gz` | Multiple demand | 20 (10 per hemisphere) | Hard > Easy spatial working memory (H-E) | `MD_all.nii.gz`, copied unchanged |
| `TOM.nii.gz` | Theory of mind | 10 (5 per hemisphere) | False belief > False photo (bel-pho) | `TOM_all.nii.gz`, copied unchanged |
| `PHYSICS.nii.gz` | Intuitive physics | 11 (6 left, 5 right) | Physical > Color judgement (P-C) | `PHYSICS_all.nii.gz`, copied unchanged |
| `PHYSICS_Kean.nii.gz` | Intuitive physics, alternative | 10 (6 left, 4 right) | Physical > Color judgement (P-C, TowerLoc) | Hope Kean's `n40_TowerLoc_P_C_fROIs_0.6percentSUB_100minVOXELS_core.nii`, gzipped unchanged |

Source: Colton Casto's exportation project (Engaging: `/orcd/archive/evelina9/001/u/ccasto/projects/exportation/code/parcels/`), designated as the final parcels, except `PHYSICS_Kean.nii.gz`, which is the physics parcellation used earlier in WinoGrande (from 40 participants' TowerLoc maps, built in this space). `checksums.tsv` gives the sha256 of every file here and of its source file. They are identical for the copied files; for `PHYSICS_Kean.nii.gz` the source checksum is that of the uncompressed content.

## Labels

Each `<NETWORK>_labels.tsv` has one row per parcel: `label` (the voxel value), `name`, `hemisphere`, `n_voxels`, and the centroid in MNI mm (`centroid_x`, `centroid_y`, `centroid_z`; negative x = left). Names and label order are those of the source `<NETWORK>_labels.txt` files, where line n names label n. The Kean physics file comes without names and its labels are 1–4, 6, 7, 9–11 and 16. So `PHYSICS_Kean_labels.tsv` has no `name` column; instead it gives, for each label, the parcel of `PHYSICS.nii.gz` it overlaps most (`closest_casto_parcel`) and the share of its voxels inside it (`share_in_closest`). That is a spatial correspondence, not a name.

```python
import nibabel as nib, numpy as np, pandas as pd
img = nib.load("TOM.nii.gz")
lab = np.round(np.asanyarray(img.dataobj)).astype(int)   # round: MD is stored as float with rounding noise
names = pd.read_csv("TOM_labels.tsv", sep="\t")
mask_rtpj = lab == names.set_index("name").loc["RH_TPJ", "label"]
```

## Derivation of the language file

`LANGUAGE_noAngG.nii.gz` is `LANGUAGE_all.nii.gz` with the two angular-gyrus parcels (label 6, LH_AngG; label 12, RH_AngG) set to 0. Every other voxel keeps its original value, so the labels run 1–5 and 7–11 and match the source names. The angular gyrus is left out of the language network in these analyses. The source file with all 12 parcels is identical, voxel for voxel, to the 2020 lab parcels (`ROIS_Nov2020/Func_Lang_LHRH_SN220/allParcels_language.nii`).

## Notes and known differences from older parcel files

- **ToM** is identical, voxel for voxel, to `ROIS_Nov2020/Func_ToM_LHRH_FBFP400/allParcels_ToM.nii`.
- **MD** matches the 2017 parcels (`MDfuncparcels_Apr2017`), moved onto the 91 × 109 × 91 grid, for all parcels except the right precentral pair. Here RH_Precentral_A_PrecG / RH_Precental_B_IFGop have 1180 / 631 voxels, mirroring the left hemisphere exactly. The 2017 file has 955 / 856, with the same 1811 voxels in total. So about 225 voxels on the boundary between the two right precentral parcels carry the other label; 99.2% of MD voxels agree. The spelling "Precental" is kept from the source label list.
- **Physics** replaces the TowerLoc parcels used earlier (`n40_TowerLoc_P_C_fROIs_0.6percentSUB_100minVOXELS_core.nii`, 10 labels among 1–16). The two sets are different parcellations: Dice overlap 0.57 over the whole network, and label numbers do not correspond. The physics parcels were drawn on fsaverage and brought into this space with an affine registration of the fsaverage template to the SPM template (FreeSurfer `mri_robust_register`, then `mri_vol2vol --nearest`). A header-only transform of the same file gives Dice 0.83 with this one, so the registration moves voxels appreciably. The right hemisphere has no TO parcel (lTO has no counterpart).
- **The two physics parcellations** (`PHYSICS.nii.gz` vs `PHYSICS_Kean.nii.gz`) cover the same dorsal frontal and parietal regions, but no parcel corresponds one to one. At best 32–83% of a Kean parcel lies in its closest counterpart. Kean splits left MPL (labels 6 and 9) and left DMFC (labels 7 and 16) in two. Kean has nothing corresponding to the occipital DO and TO parcels (lDO, rDO, lTO). Kean label 2 (left superior parietal) has 14 voxels on the midline and 4 just across it.
- **Overlap between networks.** The files are independent, and some voxels belong to more than one network. The largest overlap is MD–physics: 6684 voxels, about 64% of the physics parcels. The others are language–ToM 3513, language–MD 1392, ToM–physics 854, MD–ToM 844 and language–physics 750. fROIs defined by each subject's own localizer contrast are what separate the networks.

