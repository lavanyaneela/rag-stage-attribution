# Deployment Freeze Windows

A deployment freeze suspends all non emergency production deploys for a named window. The standing freeze windows are the last five calendar days of each fiscal quarter and the published holiday freeze in late December. Freeze windows are published in the engineering calendar at least 30 days in advance.

During a freeze, emergency deploys require sign off from the on-call director and must reference an active incident. A fix for a SEV1 or SEV2 always qualifies as an emergency. Routine dependency bumps and feature flags default to off do not qualify.

Freezes apply to production environments only. Staging and development environments deploy normally during a freeze. Database schema migrations are frozen even in staging during the quarterly windows because staging restores feed production drills.

Freeze exceptions for planned launches are requested at least 10 working days before the window starts. Exception requests name the rollback plan and the single accountable engineer. Marketing commitments are explicitly not grounds for an exception by themselves.

The freeze calendar is owned by release engineering. A deploy attempted during a freeze without an exception is blocked by the pipeline, and the attempt is logged and reported weekly. Repeated blocked attempts by the same service trigger a conversation with the owning director.
