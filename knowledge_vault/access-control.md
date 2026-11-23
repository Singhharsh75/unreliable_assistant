---
id: policy-access-control
domain: governance
risk_tier: 2
version: 1.0
---
# Access Control and Permission Tiers

System actions are grouped into three tiers.
TIER_1_READ: read records and search the knowledge base.
TIER_2_COMPUTE: run calculations and reports that do not change records.
TIER_3_MUTATE: change records, such as lot release, status changes, and form submission. Tier 3 always needs human approval, see [[policy-hitl-approval]].

Access is granted by role, never per person. Role assignments are reviewed every quarter by each department head.
Accounts are disabled within 24 hours of a person leaving the company. Accounts unused for 90 days are disabled automatically.
Every request carries a tenant ID, and a user can only see data for their own tenant.
