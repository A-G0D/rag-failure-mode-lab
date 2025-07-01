"""Build the made-up Stillwater Industrial ERP corpus and query set. Three
workflows (order-to-cash, procure-to-pay, inventory counts) plus queries with
gold answers and stress labels. Writes to docs/datasets/ and mirrors into
tests/fixtures/. All names/emails/URLs are invented.

    python scripts/build_dataset.py
"""
from __future__ import annotations

import json
from pathlib import Path

# entries are (doc_id, source, domain_tags, text)
CORPUS: list[tuple[str, str, list[str], str]] = [
    # order-to-cash
    ("DOC_O2C_01", "o2c/sales_order_entry.md", ["order_to_cash", "sales"],
     "The Order Entry Module records a customer sales order. Each sales order "
     "must reference a valid customer account and at least one line item. A "
     "sales order moves through the states draft, confirmed, allocated, and "
     "shipped. Stillwater Industrial policy requires a credit check before a "
     "sales order is confirmed."),
    ("DOC_O2C_02", "o2c/credit_check.md", ["order_to_cash", "credit"],
     "A credit check compares the customer outstanding balance plus the new "
     "order total against the customer credit limit. If the projected balance "
     "exceeds the credit limit, the order is placed on credit hold and routed "
     "to a credit analyst for manual review. Orders under the limit are "
     "approved automatically."),
    ("DOC_O2C_03", "o2c/allocation.md", ["order_to_cash", "inventory"],
     "Allocation reserves on-hand inventory against confirmed sales order "
     "lines. The allocation engine prefers the warehouse nearest the ship-to "
     "address. If on-hand quantity is insufficient, the line is partially "
     "allocated and a backorder record is created for the remainder."),
    ("DOC_O2C_04", "o2c/shipping.md", ["order_to_cash", "logistics"],
     "Shipping confirms that allocated goods have left the warehouse. A pick "
     "list is generated, items are packed, and a shipment is posted. Posting a "
     "shipment decrements on-hand inventory and triggers invoice generation in "
     "the billing step."),
    ("DOC_O2C_05", "o2c/invoicing.md", ["order_to_cash", "billing"],
     "Invoicing produces a customer invoice from a posted shipment. The invoice "
     "total equals the sum of shipped line amounts plus tax and freight. "
     "Payment terms default to net 30 days. An unpaid invoice past its due date "
     "becomes a delinquent receivable."),
    ("DOC_O2C_06", "o2c/cash_application.md", ["order_to_cash", "accounts_receivable"],
     "Cash application matches an incoming customer payment to one or more open "
     "invoices. A payment that fully settles an invoice closes it. A short "
     "payment leaves the invoice partially open. Unapplied cash is held in a "
     "suspense account until a clerk reconciles it."),

    # procure-to-pay
    ("DOC_P2P_01", "p2p/requisition.md", ["procure_to_pay", "purchasing"],
     "A purchase requisition is an internal request to buy goods or services. "
     "The Purchasing Module routes a requisition for approval based on the "
     "requested amount. Requisitions above the department threshold require "
     "manager approval before a purchase order is created."),
    ("DOC_P2P_02", "p2p/purchase_order.md", ["procure_to_pay", "purchasing"],
     "A purchase order is a binding commitment to a supplier. It lists item, "
     "quantity, unit price, and expected delivery date. A purchase order is "
     "issued only after the underlying requisition is approved. Changes after "
     "issue require a purchase order revision."),
    ("DOC_P2P_03", "p2p/goods_receipt.md", ["procure_to_pay", "inventory"],
     "Goods receipt records the arrival of items against a purchase order. The "
     "receiving clerk verifies received quantity against ordered quantity. A "
     "quantity variance beyond tolerance is flagged for review. Receipt "
     "increases on-hand inventory for the receiving warehouse."),
    ("DOC_P2P_04", "p2p/invoice_matching.md", ["procure_to_pay", "accounts_payable"],
     "Three-way matching compares the supplier invoice against the purchase "
     "order and the goods receipt. The invoice is approved for payment only "
     "when price, quantity, and receipt all agree within tolerance. A mismatch "
     "places the invoice on hold for the accounts payable team."),
    ("DOC_P2P_05", "p2p/payment_run.md", ["procure_to_pay", "accounts_payable"],
     "A payment run selects approved supplier invoices that are due and "
     "generates payments. The Accounts Payable Module groups payments by "
     "supplier and payment method. Early-payment discounts are taken when the "
     "discount date has not passed."),
    ("DOC_P2P_06", "p2p/supplier_master.md", ["procure_to_pay", "master_data"],
     "The supplier master holds supplier name, remittance address, payment "
     "terms, and tax identifier. A supplier must be active in the supplier "
     "master before a purchase order can be issued. Banking details require "
     "dual approval to change."),

    # inventory counts
    ("DOC_INV_01", "inventory/cycle_count.md", ["inventory_counts", "inventory"],
     "A cycle count audits a subset of inventory locations on a rolling "
     "schedule without halting operations. High-value items are counted more "
     "frequently. The Inventory Module compares the counted quantity against "
     "the system on-hand quantity to compute a count variance."),
    ("DOC_INV_02", "inventory/physical_count.md", ["inventory_counts", "inventory"],
     "A physical count freezes transactions and counts every location in a "
     "warehouse. It is typically performed at fiscal year end. Counted "
     "quantities are entered against count tags, and the system posts "
     "adjustments for any difference from on-hand records."),
    ("DOC_INV_03", "inventory/variance_adjustment.md", ["inventory_counts", "accounting"],
     "A count variance posts an inventory adjustment. A positive variance "
     "increases on-hand quantity and credits an inventory gain account. A "
     "negative variance decreases on-hand quantity and debits a shrinkage "
     "account. Adjustments above a value threshold require supervisor approval."),
    ("DOC_INV_04", "inventory/abc_classification.md", ["inventory_counts", "planning"],
     "ABC classification ranks items by annual consumption value. Class A items "
     "are the highest-value minority and are counted most often. Class C items "
     "are low value and counted least often. The classification drives the "
     "cycle-count frequency for each item."),
    ("DOC_INV_05", "inventory/lot_tracking.md", ["inventory_counts", "traceability"],
     "Lot tracking assigns a lot number to received goods so that on-hand "
     "quantity can be traced by lot. During a count, each lot is counted "
     "separately. Lot tracking supports recall handling by identifying which "
     "shipments contained a given lot."),
    ("DOC_INV_06", "inventory/safety_stock.md", ["inventory_counts", "planning"],
     "Safety stock is the buffer quantity held to absorb demand variability. "
     "When on-hand quantity falls below the reorder point, the planning engine "
     "proposes a replenishment order. Safety stock does not by itself trigger a "
     "purchase requisition; the reorder point does."),

    # cross-cutting architecture docs, also serve as long-context filler
    ("DOC_ARC_01", "architecture/modules_overview.md", ["architecture"],
     "The Stillwater ERP platform is organized into modules: Order Entry, Purchasing, "
     "Inventory, Accounts Receivable, and Accounts Payable. Modules share a "
     "common master-data layer for customers, suppliers, and items. Each module "
     "writes to a shared transaction ledger."),
    ("DOC_ARC_02", "architecture/master_data.md", ["architecture", "master_data"],
     "Master data is the slowly changing reference data shared across modules: "
     "customer master, supplier master, and item master. Transactional records "
     "such as orders and invoices reference master-data keys. Master data is "
     "governed centrally to avoid duplication."),
    ("DOC_ARC_03", "architecture/audit_trail.md", ["architecture", "compliance"],
     "Every posting writes an audit-trail entry capturing the user, timestamp, "
     "and before/after values. The audit trail is append-only and cannot be "
     "edited. Auditors query the trail to reconstruct how a balance was reached."),
    ("DOC_ARC_04", "architecture/period_close.md", ["architecture", "accounting"],
     "Period close locks an accounting period so no further postings can be "
     "made to it. Sub-ledgers for receivables and payables are reconciled to "
     "the general ledger before the period is closed. A closed period can be "
     "reopened only with controller approval."),
]


# entries are (query_id, text, stress_type, gold_answer, expected_context_ids)
QUERIES: list[tuple[str, str, str, str, list[str]]] = [
    # normal
    ("Q_N_01", "What must happen before a sales order is confirmed?",
     "normal",
     "A credit check must be completed before a sales order is confirmed.",
     ["DOC_O2C_01", "DOC_O2C_02"]),
    ("Q_N_02", "How does three-way matching decide to approve a supplier invoice?",
     "normal",
     "Three-way matching approves the invoice only when price, quantity, and "
     "goods receipt all agree within tolerance.",
     ["DOC_P2P_04"]),
    ("Q_N_03", "What triggers a backorder during allocation?",
     "normal",
     "Insufficient on-hand quantity causes a partial allocation and creates a "
     "backorder for the remainder.",
     ["DOC_O2C_03"]),
    ("Q_N_04", "What does a positive inventory count variance do to the accounts?",
     "normal",
     "A positive variance increases on-hand quantity and credits an inventory "
     "gain account.",
     ["DOC_INV_03"]),
    ("Q_N_05", "When are early-payment discounts taken in a payment run?",
     "normal",
     "Early-payment discounts are taken when the discount date has not passed.",
     ["DOC_P2P_05"]),
    ("Q_N_06", "What are the default payment terms on a customer invoice?",
     "normal",
     "Payment terms default to net 30 days.",
     ["DOC_O2C_05"]),

    # adversarial: invites a confident wrong answer
    ("Q_A_01", "What is the exact dollar credit limit Stillwater applies to every customer?",
     "adversarial",
     "The corpus does not state a fixed dollar credit limit; the limit is "
     "per-customer and compared against the projected balance.",
     ["DOC_O2C_02"]),
    ("Q_A_02", "Does safety stock automatically create a purchase requisition?",
     "adversarial",
     "No; safety stock does not trigger a requisition. The reorder point does.",
     ["DOC_INV_06"]),
    ("Q_A_03", "Can a closed accounting period be reopened by any clerk?",
     "adversarial",
     "No; a closed period can be reopened only with controller approval.",
     ["DOC_ARC_04"]),

    # ambiguous: more than one defensible reading
    ("Q_M_01", "What is a count in the inventory module?",
     "ambiguous",
     "It can mean a cycle count, which audits a subset of locations on a "
     "rolling schedule, or a physical count, which freezes transactions and "
     "counts every location.",
     ["DOC_INV_01", "DOC_INV_02"]),
    ("Q_M_02", "How is a payment handled?",
     "ambiguous",
     "A customer payment is matched to open invoices in cash application, while "
     "a supplier payment is generated by a payment run for approved invoices.",
     ["DOC_O2C_06", "DOC_P2P_05"]),
    ("Q_M_03", "What does approval control here?",
     "ambiguous",
     "Approval can mean requisition approval before a purchase order, or "
     "supervisor approval for large inventory adjustments, or controller "
     "approval to reopen a period.",
     ["DOC_P2P_01", "DOC_INV_03", "DOC_ARC_04"]),

    # long_context_overload: the answer is a needle in the filler
    ("Q_L_01", "Which step decrements on-hand inventory and triggers invoice generation?",
     "long_context_overload",
     "Posting a shipment decrements on-hand inventory and triggers invoice "
     "generation.",
     ["DOC_O2C_04"]),
    ("Q_L_02", "What record captures the user, timestamp, and before/after values of every posting?",
     "long_context_overload",
     "The audit-trail entry captures the user, timestamp, and before/after "
     "values of every posting.",
     ["DOC_ARC_03"]),
    ("Q_L_03", "Which classification drives how often each item is cycle counted?",
     "long_context_overload",
     "ABC classification drives the cycle-count frequency for each item.",
     ["DOC_INV_04"]),
]


def build() -> tuple[list[dict], list[dict]]:
    docs = [
        {"doc_id": d, "text": t, "source": s, "domain_tags": tags}
        for (d, s, tags, t) in CORPUS
    ]
    queries = [
        {
            "query_id": qid,
            "text": qtext,
            "stress_type": stype,
            "gold_answer": gold,
            "expected_context_ids": ctx,
        }
        for (qid, qtext, stype, gold, ctx) in QUERIES
    ]
    return docs, queries


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    docs, queries = build()
    targets = [root / "docs" / "datasets", root / "tests" / "fixtures"]
    for d in targets:
        _write_jsonl(d / "erp_corpus.jsonl", docs)
        _write_jsonl(d / "erp_queries.jsonl", queries)
    print(f"wrote {len(docs)} documents and {len(queries)} queries to {len(targets)} locations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
