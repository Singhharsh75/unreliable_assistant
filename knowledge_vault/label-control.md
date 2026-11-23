---
id: sop-label-control
domain: manufacturing
risk_tier: 2
version: 1.0
---
# Label Control

Printed labels are controlled materials. They are issued to the line by count and every unused label is counted back at the end of the lot.

The label reconciliation must balance exactly: labels issued = labels applied + labels returned + labels destroyed. Any difference is a deviation, see [[sop-deviation]].
Each label shows the product name, SKU, lot number, and expiry date. Lot numbers follow [[catalog-lot-numbering]].

Label artwork changes go through [[sop-change-control]]. Obsolete labels are destroyed under witness on the day a new version is approved.
A label mix-up found after release is a Critical deviation and is assessed for recall, see [[sop-recall]].
