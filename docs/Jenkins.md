# Jenkins CI (MIGraphX)

## Skipping tests that already passed (same commit)

When Jenkins restarts or a new build runs for the **same Git commit**, you can avoid re-running stages that already succeeded by enabling a filesystem-backed success cache.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `MIGRAPHX_CI_TEST_SUCCESS_CACHE` | Absolute path to a base directory. If **unset or empty**, caching is **disabled** and behavior matches the original pipeline. If set, each test stage writes a marker file after success and skips work on a later run when the marker exists for the same commit and image tag. |
| `MIGRAPHX_CI_FORCE_TESTS` | If `true` or `1`, ignores the cache and always runs all tests. |

### Cache key

Success is keyed by:

- Jenkins `JOB_NAME` (sanitized for paths)
- `git rev-parse HEAD` at checkout
- Docker image tag (`IMAGE_TAG` for main CI image; `IMAGE_TAG_ORT` for ONNX Runtime tests)
- A fixed stage id (e.g. `all_targets_release`, `hip_clang_release`, `onnx_runtime_tests`)

Markers are stored as:

`<cache>/<job_name>/<commit>/<image_tag>/<stage_id>.ok`

### Shared vs per-agent cache

- **Shared storage (recommended):** Point `MIGRAPHX_CI_TEST_SUCCESS_CACHE` at an NFS path (or similar) visible to **all** agents (mi100+, nogpu, Navi, onnxrt, etc.). Then a stage that passed on one machine can be skipped when another agent picks up the same commit.
- **Per-agent cache:** If the path is only local (e.g. under `/workspaces/.cache/...` on each machine), skips apply only when that **same** agent runs the same stage again for the same commit. Other agents will not see the markers.

### HIP Clang Release and ONNX Runtime

The ONNX stage consumes `.deb` packages via Jenkins **stash** from HIP Clang Release. After a controller restart, a new build has no stash. With the cache enabled:

- After a successful HIP Clang Release build, `build/*.deb` is copied to  
  `<cache>/<job_name>/debs/<commit>/<IMAGE_TAG>/`.
- If HIP Clang Release is skipped (marker hit), those `.deb` files are restored into `build/`, then **stashed** again so the ONNX agent can **unstash** as before.

**HIP skip + ONNX run requires** the cached `.deb` files to exist. If a marker exists but the deb directory is missing or empty, HIP Clang Release runs a full build again.

For ONNX-only skips, the marker uses `IMAGE_TAG_ORT` so ORT image changes invalidate the cache without a code change.

### Caveats

- **Flaky tests:** A stage that passed once may be skipped on retry; use `MIGRAPHX_CI_FORCE_TESTS=true` or delete the relevant marker/cache entries to force a re-run.
- **Cleanup:** Old cache entries are not TTL-pruned by the pipeline; remove them manually if disk use grows.
