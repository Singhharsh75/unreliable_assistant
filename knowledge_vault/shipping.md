---
id: sop-shipping
domain: logistics
risk_tier: 2
version: 1.0
---
# Shipping and Distribution

Only lots with status RELEASED can be shipped. A lot on HOLD, in QUARANTINE, or RECALLED is blocked by the warehouse system.

Orders are picked first-expiry, first-out (FEFO). A lot with less than 6 months of remaining shelf life is not shipped unless the customer agrees in writing.
Cold-chain shipments use validated insulated shippers with a temperature logger in each shipper, and must arrive within 48 hours, see [[sop-cold-chain]].

Each shipment record stores the lot number, customer, quantity, and dispatch date. These records support the 4-hour trace in [[sop-recall]] and are kept per [[policy-record-retention]].
