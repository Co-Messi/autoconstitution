# autoconstitution Demo Video Script

## Opening

**ON SCREEN**
`autoconstitution`

**VOICEOVER**
"What if autoresearch had multiple agents instead of one?"

---

## Problem

**VOICEOVER**
"Single-agent improvement loops are powerful, but they only get one perspective at a time. One model proposes a direction, but no one is assigned to attack it, challenge it, or teach it a better way."

---

## Core idea

**ON SCREEN**
Student -> Critic -> Teacher -> Judge -> Keep or Revert

**VOICEOVER**
"autoconstitution turns the autoresearch loop into a small society of agents. A student proposes. A critic attacks weaknesses. A teacher suggests a better direction. A judge decides what survives. The system keeps improvements and rejects regressions."

---

## Constitution

**ON SCREEN**
`constitution.md`

**VOICEOVER**
"The rules are explicit. They live in a Markdown file called `constitution.md`, so the behavior is inspectable, editable, and version-controlled."

---

## Demo

**VOICEOVER**
"You can run the loop locally with Ollama or plug in stronger cloud models when you need them."

**ON SCREEN**
```bash
pip install autoconstitution
ollama pull llama3.1:8b
autoconstitution cai run -p "Design a better financial-analysis workflow."
```

---

## Why it matters

**VOICEOVER**
"This pattern is broader than one domain. The same loop can improve coding agents, financial-analysis workflows, research assistants, and prompt systems. The task changes. The improvement pattern stays the same."

---

## Closing

**VOICEOVER**
"If autoresearch is one improving agent, autoconstitution is a small society of improving agents."

**ON SCREEN**
`autoconstitution`
`autoresearch, but with multiple agents`
