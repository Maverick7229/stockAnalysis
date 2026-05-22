# stockAnalysis (archived &mdash; superseded)

> **This repo is an early experiment and is no longer maintained.** It contains
> several prototype scripts (`claudeCode.py`, `claudeCode2.py`, `crewaiFA.py`,
> `financial_agent.py`) that explored different multi-agent approaches for
> stock analysis on the Indian NSE / BSE market.
>
> The work was consolidated into a cleaner project here:
>
> &nbsp;&nbsp;&nbsp;&nbsp;**[Maverick7229/inputStockName](https://github.com/Maverick7229/inputStockName)**
>
> Please use that repository for the final implementation, architecture, and
> setup instructions.

## What was explored here

- `crewaiFA.py` &mdash; an early attempt with CrewAI-style agents.
- `financial_agent.py` &mdash; a single-agent baseline using `phidata`.
- `claudeCode.py` / `claudeCode2.py` &mdash; iterative drafts during prototyping.

## Why it was archived

- Multiple competing approaches in one repo made the project hard to read.
- `inputStockName` adopted a single, cleaner LangChain-based architecture with
  a clear ticker-resolution + analyst-chain split.
