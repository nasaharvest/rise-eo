# `flatten` requires the same number of calendar windows, not the same dates

**Status:** accepted (spec 214, 2026-11-23). Supersedes [ADR 0010](0010-calendar-interval-mosaic-default.md)
in part: calendar-interval mosaics stay the default; `flatten`'s "identical `timestamps`" requirement
becomes "identical T".

**Context.** ADR 0010 gives every cube built over the same `start`/`end`/`mosaic_days` an identical
`timestamps` axis, and `flatten` refuses cubes whose axes differ. Training labels for the project's
model come from several years, so each label polygon has its own one-year window. Their cubes have
the same number of windows (T = 12) but different dates, and `flatten` refuses them. CropHarvest
(NASA Harvest, 2021) exports each label over its own `start_date`/`end_date` window in the same way.

**Decision.** `create_training_data` accepts per-polygon `startdate`/`enddate` columns. Every
polygon must give the same T for the call's single `mosaic_days`, or the call raises before any
work. `flatten` checks that every cube has the same **number** of timestamps and the same bands. It
writes `window_starts.npy`, one window start per sample, aligned with `ids.npy`, so the absolute
dates stay recoverable. Cross-cell merging at inference still uses one window for the whole ROI, so
ADR 0010's guarantee holds there unchanged.

**Options rejected.**
- **Pad or truncate to a common T.** That silently reshapes data the model will see.
- **One `create_training_data` call per window, concatenated by the user.** That pushes the
  bookkeeping (ids, labels, dates) onto every user and loses the shared skip identity.

**Consequences.**
- A training set can span years.
- Code that reads `metadata["timestamps"]` as *the* dates of every sample is wrong for a per-polygon
  set and must read `window_starts.npy` instead.
- The `AGENTS.md` line "`flatten` needs that" changes with the code.
