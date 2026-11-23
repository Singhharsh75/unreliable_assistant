---
id: policy-audit-trail
domain: governance
risk_tier: 2
version: 1.0
---
# Audit Trail

Every GMP computer system records an audit trail entry for each create, change, and delete. An entry holds who, what, when (UTC timestamp), the old value, the new value, and the reason.

Audit trails cannot be switched off or edited by any user, including administrators.
QA reviews the audit trail of each lot's electronic records as part of the batch record review, see [[sop-batch-record-review]].
System administrators' activity is reviewed separately every month.

The assistant writes an audit event for every tool call, including refused and expired Tier 3 requests.
Audit trails are kept as long as the records they describe, see [[policy-record-retention]].
