# 03: Report data problems

**What to build:** The audit reports missing values, duplicates, constant columns, conflicting labels and invalid labels, reusing the guard library's profiling and streaming.

**Blocked by:** 01

**Status:** done (v0.6-audit)

- [x] One planted-defect fixture per problem produces exactly that finding
- [x] A clean control produces no critical or high finding
- [x] A file larger than the sampling limit is handled by streaming, and any sampled check states its sample size
- [x] The report contains column names and counts, and a test confirms no raw row values appear
