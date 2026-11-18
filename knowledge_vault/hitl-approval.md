---
id: policy-hitl-approval
domain: governance
risk_tier: 3
version: 1.0
---
# Human Approval for Tier 3 Actions

Any action that changes records (lot release, status change, form submission) is Tier 3.
The assistant never executes Tier 3 actions itself. It creates a PENDING_APPROVAL request and
an authorized reviewer approves or rejects it. Requests not decided within 24 hours expire as REJECTED.
