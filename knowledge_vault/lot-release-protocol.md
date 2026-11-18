---
id: sop-lot-release
domain: quality
risk_tier: 2
version: 1.0
---
# Lot Release Safety Protocol

Every production lot (format LOT-NNNNN) must pass a safety verification before release.
Verification requires: (1) batch record review complete, (2) all in-process checks within limits,
(3) no open deviations against the lot, and (4) QA sign-off recorded in the state store.

A lot in status HOLD cannot be shipped. Releasing a lot is a Tier 3 mutation and requires
reviewer approval, see [[policy-hitl-approval]]. Open deviations are handled per [[sop-deviation]].
Cold-chain lots also need a temperature log check, see [[sop-cold-chain]].
