# Remote Access and VPN

Remote access to internal systems requires the corporate VPN with device certificates issued by IT. Certificates expire every 180 days and renew automatically on managed laptops. A device without a valid certificate cannot reach internal DNS at all.

VPN sessions disconnect after 12 hours and require fresh multi factor authentication to reconnect. Long running jobs must run on servers, not on laptops over VPN. Session length is a security boundary, not a convenience setting.

Personal devices are never granted VPN certificates. Contractors receive certificates limited to 90 days tied to their contract end date. Certificate scope limits which internal zones a device can reach, and production zones require a separate just in time grant.

Split tunneling is disabled; all traffic from a connected device routes through the VPN. The exception list is owned by Security and currently contains only the major video conferencing services. Requests to add an exception require a data flow review.

Access from countries on the restricted list is blocked at the VPN edge. Travel to a restricted country requires a loaner laptop issued by IT and a pre travel briefing. Loaner laptops are wiped within 48 hours of return.
