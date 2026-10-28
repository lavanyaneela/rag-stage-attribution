# Password and Authentication Policy

Workforce authentication is through single sign on with hardware backed passkeys as the primary factor. Passwords still exist for the identity provider account itself and must be at least 16 characters. Password rotation is not required unless compromise is suspected, in line with current guidance.

Multi factor authentication is mandatory for every workforce account with no exceptions. Push fatigue attacks are mitigated by number matching on every push approval. SMS codes are not an accepted factor anywhere in the workforce stack.

Shared service accounts are inventoried, vaulted, and checked out per use. Checking out a vaulted credential records who, when, and for which change ticket. Static shared passwords in team wikis are a finding in every internal audit and are removed on sight.

Personal password managers are provided to all employees with a family plan included. Storing work credentials in a personal browser profile is prohibited. The provided manager is the only approved store for work secrets that cannot live in the vault.

Suspected credential compromise is reported to Security immediately and resets are forced within one hour of the report. Sessions are revoked globally at reset, which signs the user out of every device. Reporting a suspected compromise is never a disciplinary matter, including when the suspicion proves wrong.
