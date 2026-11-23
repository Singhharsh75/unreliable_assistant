---
id: sop-qa-signoff
domain: quality
risk_tier: 3
version: 1.0
---
# QA Sign-Off

QA sign-off is step 4 of lot release, see [[sop-lot-release]]. Only a Qualified Person (QP) may sign off a lot.

Before signing, the QP confirms that the batch record review is complete, all in-process checks are within limits, there are no open deviations against the lot, and, for cold-chain lots, the temperature log is acceptable.
The sign-off is recorded in the state store with the QP's user ID and a timestamp. It is a Tier 3 action and goes through [[policy-hitl-approval]].

A QP may not sign off a lot they personally manufactured or reviewed in production.
A sign-off can be revoked only by the Head of Quality. Revoking a sign-off puts the lot on HOLD.
