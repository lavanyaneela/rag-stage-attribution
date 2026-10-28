# Backup and Recovery

Production databases take a full snapshot every 24 hours and incremental snapshots every 15 minutes. Snapshots are stored in a separate account from the databases they protect. Access to the snapshot account is limited to the storage platform team.

Snapshots are retained for 35 days and then deleted automatically. The 35 day window is a recovery mechanism, not a retention archive; the data retention schedule governs how long data may exist. A restore never extends the life of data past its retention line.

The recovery point objective for tier one services is 15 minutes and the recovery time objective is four hours. Tier two services carry a recovery point objective of 24 hours. Tier assignments are reviewed yearly and owned by the service's director.

A full restore drill runs twice a year for every tier one service. The drill restores into an isolated environment and replays a fixed verification suite. A drill that cannot finish within the recovery time objective opens a SEV2 against the owning team.

Backup failures page the storage platform rotation, not the owning service's rotation. Two consecutive failed snapshots of the same database is an automatic SEV2. Silent gaps are prevented by a monitor that alerts on snapshot age, not on job exit codes.
