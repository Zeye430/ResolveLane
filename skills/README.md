# Skills

Handling guides for the reply agent, maintained by support operations. Each skill is a directory with a `SKILL.md` file. The application validates every skill at startup and refuses to start if one is invalid.

```markdown
---
name: missing-item            # lowercase words joined by hyphens; must match the directory name
version: 1                    # increase whenever the guidance changes; replies log the version used
description: One line saying when the skill applies
issues: missing_item          # issue types that select the skill, comma separated
keywords: left out, not in the bag   # optional: lowercase words or phrases that also select it
enabled: true                 # optional; false keeps the file but stops using it
---
Guidance in plain sentences.
```

A skill is used when the case has raised one of its issues, or when the customer's latest message contains one of its keywords as whole words. Every skill needs at least one issue or keyword; guidance for every case belongs in the agent's fixed instructions.

Skills hold guidance, not facts. Do not write times, amounts, deadlines or policy terms in a skill: those belong in support policies and help center articles, where they are versioned and cited. A reply that repeats a number found only in a skill is rejected by the grounding check. A guide cannot override the agent's fixed rules, such as never promising a refund.

A skill's guidance is at most 1,500 characters; the guides selected for one reply are at most 4,000 characters together.
