# Incident Escalation

An incident is any unplanned event that degrades a production service for users. Incidents are classified at declaration as SEV1, SEV2, or SEV3 and the classification may change as facts arrive. Declaring too high and downgrading is preferred over declaring too low.

A SEV1 requires an incident commander within 15 minutes of declaration. The incident commander is not the person debugging; their job is coordination, communication, and deciding when to escalate further. SEV1 status updates are posted every 30 minutes until mitigation.

SEV2 incidents require a response within one hour and status updates every two hours during business hours. A SEV2 that is not mitigated within eight working hours is automatically reviewed for promotion to SEV1. SEV3 incidents are queued as high priority tickets and need no live response.

Executive notification is mandatory for any SEV1 lasting longer than 60 minutes. The notification goes to the VP of Engineering and the head of Customer Success simultaneously. Customer facing communication is owned by Customer Success, never by the responding engineers.

Every SEV1 and SEV2 gets a written postmortem within five working days of mitigation. Postmortems are blameless and name systems, not people. Action items from a postmortem are tracked to closure and reviewed at the monthly operations meeting.
