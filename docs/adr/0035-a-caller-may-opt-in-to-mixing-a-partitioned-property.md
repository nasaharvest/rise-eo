# A caller may opt in to mixing a partitioned property by naming every value to mix

**Status:** accepted (spec 214, 2026-11-23). Supersedes [ADR 0029](0029-collections-declare-what-must-not-be-mosaicked.md)
in part: the declaration and the default raise stay; the "never mix" outcome becomes a default.

**Context.** ADR 0029 makes a build raise when its rows span more than one value of a partitioned
property, because a median of ascending and descending Sentinel-1 passes "corresponds to no
observation". A model the project must serve was trained on monthly Sentinel-1 medians over both
orbit states. For that model, a single-orbit cube is the wrong input. The project needs to produce
exactly the input the model expects, even though a single-orbit cube is the better radar product.
The prior art splits:
- CropHarvest, sits and WorldCereal each keep one orbit direction.
- CDSE's Sentinel-1 monthly mosaics (2024) average several viewing geometries.

**Decision.** A partition key that `properties_filter` lists with more than one value, e.g.
`{"sat:orbit_state": ["ascending", "descending"]}`, is exempt from the single-value check. With no
value or one value, ADR 0029 applies unchanged. `properties_filter` is part of the cube's identity,
so a mixed cube never shares a path with a single-orbit one. The model bundle carries its training
lineage (spec 214 D9), and inference refuses a `properties_filter` that differs from training, so
a model trained on mixed medians cannot be served single-orbit ones, and the reverse.

**Options rejected.**
- **A new `partition_policy="mix"` on a registered collection variant.** Verbs require the catalog
  folder to be named for `collection=`, so a variant cannot read the `sentinel-1-rtc` catalog without
  a second change to the archive contract.
- **A boolean such as `mix_orbits=True`.** That puts one collection's concept into a
  collection-agnostic signature, which ADR 0029 and spec 58 D9 rejected for `orbit_state=`.
- **Separate per-orbit bands** (`vv_asc`, `vv_desc`). Physically cleaner, but not the input the
  model was trained on. It remains available later as its own decision.

**Consequences.**
- Mixing is explicit and greppable in every call that does it.
- The default still protects every caller who did not ask.
- The cost (two geometries in one median) is stated in every docstring that takes
  `properties_filter`.
- Watch: a bundle saved without training lineage skips the D9 check.
