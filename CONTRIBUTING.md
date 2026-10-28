# Contributing

The bar for a change here is the bar the repository sets for itself: every
figure a document states is produced by code that ran, and `make verify`
passes offline from a fresh clone.

## Workflow

1. Fork, branch from `main`, and run `pip install -e ".[dev]"`.
2. Make the change with its tests. Test names are sentences stating the
   property; a fix for a bug carries a test that fails without it.
3. Run `make verify`. If you changed anything measured, run
   `make experiments` and commit the regenerated outputs; the receipts
   will refuse a document that disagrees with them.
4. Open a pull request describing what changed and which numbers moved.

## Ground rules

- No em dashes in any file.
- No absolute durations in prose; the duration sweep enforces this.
- No figure typed by hand; extend `tools/collect_metrics.py` with an
  anchor instead.
- The attributor never sees the injected mode or the parametric bank;
  tests enforce the wall, and changes that weaken it will not merge.
