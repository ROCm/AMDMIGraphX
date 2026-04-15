---
applyTo: ".github/**"
---

Files under `.github/` control CI workflows, Actions configurations, and repository settings. Changes here can break builds, disable security checks, or disrupt automation for the entire project.

- Always flag changes to `.github/` files with a warning in the review summary.
- Verify that modified workflows use pinned action versions (e.g. `actions/checkout@v6`, not `actions/checkout@main`).
- Check that secrets are not exposed or referenced insecurely.
- Confirm any new or changed `if` conditions in workflow steps are logically correct and fail-safe.
