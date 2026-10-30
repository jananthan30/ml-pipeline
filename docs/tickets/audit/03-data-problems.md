# 03: Report data problems

**What to build:** The audit reports missing values, duplicates, constant columns, conflicting labels and invalid labels, reusing the guard library's profiling and streaming.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] One planted-defect fixture per problem produces exactly that finding
- [ ] A clean control produces no critical or high finding
- [ ] A file larger than the sampling limit is handled by streaming, and any sampled check states its sample size
- [ ] The report contains column names and counts, and a test confirms no raw row values appear
