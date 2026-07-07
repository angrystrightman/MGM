# ComPASS-Biome MicroCorpus-260K Audit

Data root: `/home/sunyirong/shared/sunyirong/MicroCorpus-260K`

## Key Conclusion

- Teacher embedding extraction: yes.
- Taxa reconstruction target: yes, use `MicroCorpus-260K_unnorm.pkl.data`.
- Disease/pathway validation: no direct labels or pathway matrix found.
- Metadata validation target: biome/source hierarchy.

## Task Suitability

| Task | Support | Reason |
|---|---|---|
| MGM teacher embedding extraction | yes | MicroCorpus pickle is loadable. |
| taxa reconstruction target | yes | Unnormalized corpus .data is sample x genus relative abundance. |
| biome/source classification | yes | MGnify biome hierarchy columns are available. |
| body-site classification | partial | Body-site-like labels appear only inside biome hierarchy for subsets. |
| disease prediction | no | No disease/phenotype columns. |
| leave-one-study-out validation | no | No explicit study/cohort column. |
| country/geography confounder control | no | No country/region/location column. |
| sequencing-platform confounder control | no | No platform/instrument column. |
| pathway reconstruction | no | No pathway abundance matrix. |
| cvaNMF/NMF baseline | yes | Fixed sample x genus matrix is available. |
| program-cohort stability | no | No cohort labels; use biome stability instead. |
