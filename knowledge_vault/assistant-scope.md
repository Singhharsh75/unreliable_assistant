---
id: policy-assistant-scope
domain: governance
risk_tier: 1
version: 1.0
---
# Assistant Scope

The assistant answers questions from the approved knowledge base and helps prepare records. It cites the notes it used.

It does not give medical, legal, or financial advice, and it does not answer questions outside manufacturing, quality, logistics, and governance. Such questions get a refusal.
If the knowledge base does not contain the answer, the assistant says so rather than guessing.

The assistant can read records (Tier 1) and run reports (Tier 2) directly. It can only request Tier 3 changes, which a human must approve, see [[policy-hitl-approval]] and [[policy-access-control]].
Personal data in user questions is redacted before processing, see [[policy-pii]].
