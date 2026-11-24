# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Answer one request on a Mac.

    python examples/decide.py path/to/karpo-decision-31B
    python examples/decide.py path/to/karpo-decision-12B --float16
"""

from __future__ import annotations

import argparse
import json

import torch

import karpo_apple_silicon

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("checkpoint", help="folder of a Karpo Decision Gemma-family checkpoint (12B or 31B)")
parser.add_argument("--float16", action="store_true", help="convert the backbone to float16 (faster on Apple GPUs)")
args = parser.parse_args()

karpo_apple_silicon.install()  # before the first decide()

from karpo_decision.decider import Decider  # noqa: E402

decider = Decider.from_pretrained(args.checkpoint, device="mps")
if args.float16:
    decider.ckpt.backbone.to(torch.float16)

request = {
    "state": "Order 1182: the customer asks to return a blender bought 45 days ago. Policy: returns within 30 days.",
    "questions": {
        "eligible": {"type": "noul", "instructions": "Is the return eligible under the policy?"},
        "action": {"type": "choice", "instructions": "What should the agent do?",
                   "criteria": {"refund": "Issue a full refund", "decline": "Politely decline the return",
                                "escalate": "Escalate to a supervisor"}},
    },
}
print(json.dumps(decider.decide(request), indent=2))

# Several requests in one call share forward passes, which is faster per request than one call each:
tickets = ["The parcel arrived crushed and the kettle inside does not switch on.",
           "Could you tell me whether the desk lamp comes in green?",
           "I was charged twice for the same pair of headphones."]
question = {"type": "choice", "instructions": "Which team should handle this message?",
            "criteria": {"returns": "Damaged or faulty goods", "billing": "Payments and charges",
                         "sales": "Questions before buying"}}
responses = decider.decide([{"state": ticket, "questions": {"team": question}} for ticket in tickets])
for ticket, response in zip(tickets, responses, strict=True):
    print(response["answers"]["team"]["choice"], "<-", ticket)
