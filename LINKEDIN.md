General-purpose chat models are tuned to be helpful in one turn. I put one in the planning stage of an agent—output a JSON plan `{goal, steps, stop_when}` before any tool runs—and it scheduled work I did not ask for or fired gated actions too early.

On 96 held-out prompts across six enterprise tool catalogs, the base `Qwen2.5-1.5B-Instruct` planner did wasteful sequences on 53%: extra lookups, `check_coverage` and `file_claim` before `lookup_policy`, `lookup_order` bundled with `check_inventory`, or `ask_user` when `lookup_pnr` was enough. Another 29% breached policy with empty observations—`issue_refund`, `transfer`, `change_flight`, `reset_password`—instead of the legal lookup. Supervised finetuning raises the gold string; it never contrasts the near-miss the model actually writes on that prompt.

Direct Preference Optimization teaches the contrast offline: same prompt $x$, legal plan $y^+$, deterministic near-miss $y^-$. The DPO margin $\beta[\log \pi_\theta(y^+|x)/\pi_{\mathrm{ref}}(y^+|x) - \log \pi_\theta(y^-|x)/\pi_{\mathrm{ref}}(y^-|x)]$ puts gradient on the diverging tool token because shared JSON prefixes cancel. I added LoRA ($r=16$, $\alpha=32$, 0.28% trainable on q/k/v/o) with `ref_model=None` so frozen base weights are $\pi_{\mathrm{ref}}$ when the adapter is off—540 pairs, 136 steps, 997 seconds on one Colab T4, no second model in VRAM.

Greedy held-out decode: wrong-sequence plans 51/96 → 22/96; exact match 15/96 → 35/96 (22 flips); ungrounded args 0% throughout. Policy breaches 28/96 → 35/96 even though reward accuracy was 1.000 by step 10—only 43/540 pairs were gated action vs lookup, so 26/28 policy failures stayed and nine wrong-sequence rows became policy instead.

Notebook, pairs, transition matrix: https://github.com/dPKaRo/dpo-tool-plans

#DPO #LLM #AIAgents

---

## Quotable excerpts

A chat-tuned model optimizes for one helpful turn; an agent planner must emit the minimal legal next step before anything executes.

DPO does not need a reward model or rollouts—it is binary cross-entropy on $(x, y^+, y^-)$ with the reference policy in the denominator; shared prefixes cancel, so the update lands on the tool name that diverges.

With `ref_model=None`, LoRA off is $\pi_{\mathrm{ref}}$: one T4, 540 pairs, 997 seconds—no duplicate 1.5B copy in VRAM.

Reward accuracy 1.000 by step 10 did not mean greedy plans obeyed gates: policy breaches 28/96 → 35/96 because only 43/540 pairs contrasted bare policy breach vs lookup.

Wrong-sequence waste 51/96 → 22/96 and exact match 15/96 → 35/96 on held-out prompts—the curriculum you write is the behavior you get.
