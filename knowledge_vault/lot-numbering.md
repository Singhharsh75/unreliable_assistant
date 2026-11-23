---
id: catalog-lot-numbering
domain: catalog
risk_tier: 1
version: 1.0
---
# Lot Numbering

Lot numbers use the format LOT-NNNNN: the prefix LOT-, then exactly five digits, zero padded. For example, lot 44 is written LOT-00044.

Lot numbers are assigned in sequence by the manufacturing system when the batch record is opened, and are never reused, even if the lot is rejected.
One lot is made from one SKU on one line in one continuous run. A lot that is split for packaging keeps its lot number, with suffixes -A, -B, and so on for each split.
The lot status lifecycle is: QUARANTINE → RELEASED, or QUARANTINE → REJECTED. A released lot can later move to HOLD or RECALLED. See [[sop-lot-release]].
SKU formats are defined in [[catalog-sku]].
