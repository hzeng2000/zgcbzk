---
name: plain-chat-math
description: Use when replying to this user in chat, especially for math, statistics, optimization, machine learning, or other technical content that may include formulas. The user's terminal cannot render Markdown math, so assistant chat responses must use plain-text formulas instead of Markdown/LaTeX math delimiters. Markdown/LaTeX math remains allowed when creating or editing files such as Markdown notes.
---

# Plain Chat Math

## Overview

Make chat responses readable in a terminal by writing formulas as plain text. This skill affects assistant messages to the user; it does not restrict file contents.

## Chat Response Rules

- Do not use Markdown math delimiters in chat responses: `$...$`, `$$...$$`, `\(...\)`, or `\[...\]`.
- Prefer inline plain text for short formulas, for example: `Var(X) = E[X^2] - (E[X])^2`.
- Use fenced code blocks for multi-line derivations or aligned equations.
- Use ASCII names when they improve terminal readability: `mu`, `sigma^2`, `epsilon`, `theta`, `x_bar_n`.
- Keep explanations natural. Do not over-format a simple formula if a sentence is clearer.

## File Editing Rules

- When writing or editing Markdown, LaTeX, notebooks, or documentation files, use the file's existing math style.
- Markdown math is allowed in files when it is the correct target format.
- Fix malformed Markdown math in files when requested or when it is clearly broken.

## Examples

Instead of writing Markdown math in chat, write:

```text
Weak law: P(|X_bar_n - mu| > epsilon) -> 0.
Strong law: P(lim X_bar_n = mu) = 1.
```

If editing a Markdown note, it is still fine to write display math in that file when appropriate.
