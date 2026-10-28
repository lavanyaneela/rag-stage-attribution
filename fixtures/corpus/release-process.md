# Release Process

Services release from the main branch through the shared delivery pipeline; there are no release branches. A release candidate is cut automatically when main passes the full verification suite. Every release carries an immutable version derived from the commit, never assigned by hand.

A release deploys first to the canary slice, which serves 5 percent of production traffic. The canary bakes for a minimum of 60 minutes before automatic promotion. Promotion is blocked if the canary's error rate exceeds the trailing baseline by more than 20 percent.

Rollback is a redeploy of the previous version, and it is exercised, not theoretical: every service must demonstrate a rollback in staging every quarter. Rollbacks of schema changes follow the expand and contract pattern; destructive migrations are split across at least two releases.

Release notes are generated from merged pull request titles and attached to the version record. Customer visible changes additionally require a line in the customer changelog reviewed by Product. Internal only releases skip the customer changelog but never skip the version record.

Deploys are allowed Monday through Thursday during business hours for each service's primary region. Friday deploys require director approval. The pipeline enforces the calendar, including freeze windows, so policy and tooling cannot drift apart.
