# Appendix F — The Learner's Playbook

**Read once now. Then use it like Appendix A — find your symptom, work the causes.**

Appendix A debugs models. This debugs the person running them.

**More attempts at this path end for reasons in this appendix than for reasons in any other chapter.** Not because people can't do the maths. Because the maths was never the binding constraint.

---

## F.0 How to use this

Same structure as Appendix A: symptom → likely cause → action. Ranked by frequency, not by how interesting they are.

**One honest boundary first.** This appendix is about the ordinary difficulty of hard self-directed study — flat motivation, lost confidence, weeks where nothing works. That's normal and this is the right tool for it.

It is *not* a substitute for real support if what you're experiencing is more than that. If low mood, hopelessness, or exhaustion persists across weeks regardless of what you're studying, follows you into things you normally enjoy, or has you thinking you'd be better off not being here — that's not a study problem and no productivity technique addresses it. Talk to a doctor or someone you trust. Please treat that as seriously as you'd treat a hardware failure: it's not a character issue, and it's fixable with the right help.

Everything below assumes the ordinary kind.

---

## F.1 Motivation

### "I don't want to start today"

**This is almost never a motivation problem. It's an activation-energy problem.**

Motivation is not a prerequisite for starting; it's usually a *consequence* of having started. Waiting to feel like it is waiting for the wrong thing.

**Action — the ten-minute rule.** Commit to ten minutes. Set a timer. When it goes off you may stop with no guilt. You will usually keep going, because starting was the whole barrier. On the days you do stop, ten minutes still beats zero.

**Action — reduce the decision.** "Study AI" is not a task; it's a category, and your brain treats it as one big undifferentiated obligation. "Open `ch05.md`, do exercise 4" is a task. **End every session by writing tomorrow's first task**, so tomorrow you start instead of deciding.

### "This is boring"

**Check your ratio first.** Boredom is almost always a symptom of watching instead of building. Passive input is boring; building is not.

**Action:** if you've watched more than you've typed in the last three days, invert it today. Skip to the exercise. Come back to the video only when you're stuck on something specific.

### "I've lost the thread of why I'm doing this"

**Action:** reread your progress log from month one (§0.7). Not the recent entries — the earliest ones. Read what you couldn't do then.

This is the single reason the log exists. Progress is invisible from inside because your sense of "normal" moves with you. The log is the only external record that it moved.

### "What's the point, models will do this anyway"

Worth answering rather than dismissing. Two things:

The people best positioned in a field where AI does the work are the ones who understand what it's doing — and can tell when it's wrong. That understanding is exactly what you're building.

And more practically: this argument arrives most often around month four, dressed as a strategic insight. Notice when it shows up. If it appeared the same week the work got hard, it's probably the wall (§F.9) wearing a philosophical costume.

---

## F.2 Confidence

### "I'm too stupid for this"

**The single most common cause: you're comparing your process to someone else's output.**

You see finished derivations, clean repos, confident blog posts. You experience your own confusion, false starts, and the hour you lost to a shape error. That comparison is invalid and it is *structurally* invalid — the process is never what gets published.

**Action:** reread Chapter 5½. That chapter exists specifically for this. The wrong turns in it are real.

**Action:** write your own transcript for whatever you're currently stuck on. Seeing your reasoning laid out usually shows convergence you couldn't feel.

### "Everyone else learns faster"

You are comparing against a filtered sample. People who found it hard and quit are invisible. People who found it hard and continued don't post about it. What you see is the tail.

**Also:** most people who look fast had a head start you can't see — a maths degree, a job that overlapped, five years of programming. Speed differences at the start are mostly about prior exposure, and prior exposure stops mattering by about month six.

### "I'll never catch up"

**There is no finish line to catch up to.** The field's frontier moves, everyone is behind it, including the people at the frontier — who are behind it in every subfield but their own.

The relevant question isn't "am I caught up" but "can I do useful work." Those are very different bars, and the second one is much closer than it looks. Chapter 13's ablation ladder, rung 1, is genuinely reachable within a year.

### Imposter feelings generally

Near-universal in this field, including among people whose work you'll read this year. It doesn't go away with competence; it recalibrates. Expect it as background noise rather than as information.

**One useful reframe:** the feeling tracks *how much you can see that you don't know*, which grows as you learn more. It's a side effect of expanding your map, not evidence about your ability.

---

## F.3 Focus

| Symptom | Likely cause | Action |
|---|---|---|
| Can't concentrate at all | Sleep debt | Fix sleep first. Nothing else works without it. |
| Reading the same paragraph repeatedly | Task too vague | Convert to a concrete action: "implement X," not "understand X" |
| Constant tab-switching | Task too hard, avoidance response | Shrink it (Unstuck step 3). Smaller is not cheating. |
| Rabbit-holing for hours | Curiosity without a boundary | Write it in `PARKED.md`, return Sunday |
| Everything feels urgent | No plan for the day | Write three tasks in the morning. Only three. |
| Productive on easy things only | Avoiding the hard thing | Do the hard thing in block B, first, before anything else |

**On environment:** phone in another room, notifications off, one tab. This sounds trivial and produces a larger effect than any study technique. The cost of a single interruption is roughly 20 minutes of re-immersion, so three interruptions per hour means you have no focused hours.

---

## F.4 Time

### "I don't have five hours today"

**Then have ninety minutes.** From §0.7: on a bad day, do **block B (build, 90 min) and block D (Anki, 15 min)**. That's a real day. Protecting this minimum is how you survive month four.

**Do not skip entirely because you can't do the full amount.** That's the pattern that turns one missed day into a missed month.

### "I missed three days"

**The streak is not the point. The total is the point.**

Missing three days out of 365 costs you roughly 1% of the year. Treating it as failure and stopping costs you the other 99%. These are not close.

**Action:** resume today at the minimum viable day. Do not attempt to "catch up" — that's how missed days become an insurmountable debt in your head.

### "Life happened and I lost a month"

**Declare a planned pause rather than drifting.** The difference is enormous psychologically and it's mostly bookkeeping:

```markdown
## PAUSE
Paused: 2026-11-03
Reason: <whatever it is>
Resume date: 2026-11-24
Resume at: Chapter 7, exercise 9
First task back: re-do Chapter 6 checkpoint cold
```

A pause with a resume date is a decision. Drift is an ending that hasn't been acknowledged yet. Same calendar, completely different outcome.

### "I keep running out of time in the day"

Front-load block B. Study *before* the day fills up, not with whatever's left over. Whatever you schedule last never happens.

---

## F.5 The restart trap

**The symptom:** you're in Chapter 7, it's hard, and you feel a strong pull to go back and "properly relearn" linear algebra.

**Why it's a trap:** restarting is emotionally satisfying — the early material is easy now, you feel competent, progress feels fast. It's also mostly worthless. You're re-covering ground you already have while the actual gap goes unaddressed.

**Most people who fail at self-directed technical study fail this way**, not by quitting. They restart three times and never get past month three.

**How to tell a real gap from the restart urge:**

| Real gap | Restart urge |
|---|---|
| You can name the specific thing you don't know | You feel generally shaky |
| Chapter 2's checkpoint fails cold | You just feel like you'd fail it |
| Going back fixes the current blocker | Going back feels reassuring |

**Action:** don't restart. **Take the earlier chapter's checkpoint cold, right now.** If you pass it, the gap isn't there and the urge is anxiety — push forward. If you fail one specific item, fix that one item, which takes a day, not a month.

This is Unstuck step 2 (§0.3) doing more work than it appears to.

---

## F.6 Comparison

**The only valid comparison is to yourself 30 days ago.** Everything else is noise, and this is why the progress log exists.

**Social media specifically:** it is a highlight reel with a strong selection effect toward people who post. It is useful for noticing what the field is paying attention to, and actively harmful as a measure of your own progress. If it's costing you more than it gives, cut it during study hours.

**People five years ahead** are five years ahead. That's the whole explanation. You will be five years ahead of someone in five years.

---

## F.7 Sustainability

These end more attempts than any concept in this book. They're boring, which is exactly why they get ignored until they're a problem.

| Issue | Prevention | If it's already happening |
|---|---|---|
| **Wrist / RSI** | Neutral wrist position, breaks every 45 min | Stop immediately. This gets chronic fast. See a doctor. |
| **Eye strain** | 20-20-20: every 20 min, look 20 feet away for 20 seconds | Reduce screen hours; check your prescription |
| **Back / neck** | Screen at eye level, feet flat, get up hourly | Same, plus movement |
| **Sleep debt** | Non-negotiable. It's a *learning* input, not a luxury | Fix before anything else; nothing works around it |
| **No movement** | 30 min of walking daily, minimum | Add it back today |
| **Isolation** | See §F.8 | Same |

**On sleep specifically:** memory consolidation happens during sleep. Studying eight hours on five hours of sleep genuinely retains less than studying five hours on eight hours of sleep. It isn't discipline versus laziness; it's an input to the process you're running.

---

## F.8 Isolation

**Working alone for a year is the most common structural reason this fails**, more than any technical difficulty.

You need three things, and they're different:

1. **Someone at your level.** To be confused alongside. This is the one that most protects against "I'm too stupid" — seeing someone comparable struggle with the same thing is immediate, concrete evidence that the difficulty is in the material.
2. **Someone ahead of you.** For calibration. Not to teach you — just to occasionally confirm that what you're experiencing is normal.
3. **Somewhere to publish.** Even to a small audience. Writing for a reader is different from writing for yourself.

**Action, this week, not month six:** post in one community in your area (§C.9). Answer one beginner question. That's it. Being known as someone who shows up is cheap and compounds.

**On local:** in-person contact does something that text doesn't. Monthly is enough.

---

## F.9 The crisis calendar

From §0.8, expanded. **These are scheduled events, not personal failures.** Knowing the schedule removes most of their force.

| When | What it feels like | What's actually happening | Action decided in advance |
|---|---|---|---|
| **Week 3** | "I'm too stupid for maths" | First contact with real mathematical confusion. School maths was pre-digested; this isn't | Slow 20%, keep going, use the Unstuck Protocol |
| **Week 8** | "I'm just copying tutorials" | Input/build ratio has drifted | Invert it today. Build first, watch only when stuck |
| **Month 4 — The Wall** | Novelty gone, finish line invisible, feels like an unpaid job | The hardest point. Most attempts end here | **Cut to 4 hours/day for two weeks. Do not stop entirely.** Reread month-one log |
| **Month 6** | "Someone already did this better" | True, permanently, for everyone | The goal isn't to be first. It's to be able to do the work |
| **Month 9** | "My research idea is stupid" | Probably. Most are | Run the experiment anyway. A clean negative result is real (§15.6) |
| **Any time** | Comparison to people 5 years ahead | Invalid comparison | Progress log, month one |

**Write these on the same index card as the Unstuck Protocol.** When one arrives, recognizing it as scheduled is most of the remedy.

---

## F.10 Using AI without outsourcing the thinking

This is the section that didn't need to exist five years ago and now matters more than most of the rest.

### The problem, stated precisely

An AI assistant can produce the artifact you were supposed to produce. When that happens you get the artifact and lose the learning — and **the failure is invisible.** Your repo looks identical either way. Your commit history looks identical. Nobody can tell from outside, including you, for months.

That's what makes it dangerous. Every other failure mode in this appendix announces itself. This one doesn't.

### The test

> **Could I do this again tomorrow, from a blank file, without help?**

If no — and it's something you should be able to do — you outsourced it. That's it. That's the whole test, and it's the cold rebuild (§0.2) applied to your use of assistance.

### The gradient of asks

Not all asking is equal. Roughly from safe to dangerous:

| Ask | Safety | Notes |
|---|---|---|
| "Explain why X works" | ✅ Safe | Conceptual explanation is what teachers do |
| "Grade my derivation" | ✅ **Underused** | You did the work; they check it. Excellent use |
| "What would a reviewer object to?" | ✅ **Underused** | Adversarial review of your own work |
| "Give me a harder version of this problem" | ✅ Safe | Generating practice is a great use |
| "Here's my code and the error, here's what I tried" | ✅ Safe *after* Unstuck steps 1–7 | The format itself requires you to have worked |
| "What category of thing am I missing?" | ⚠️ Hint-level | Fine when stuck. Ask for a nudge, not an answer |
| "Show me how to implement X" | ⚠️ Depends | Fine if X isn't the exercise. Fatal if it is |
| "Write the autograd engine" | ❌ Never | That's Chapter 5. That's the whole point |
| "Do exercise 7" | ❌ Never | — |

### Six rules

**1. Never ask for code you haven't attempted.** Attempt first, badly, for at least 30 minutes. The attempt is where the learning is; the answer is just the answer.

**2. Ask for hints, not answers.** "What class of thing am I missing?" gets you unstuck without removing the work. "Am I on the right track with X?" is better than "what's the answer."

**3. Close everything and redo it.** After any substantive help, close the window, open a blank file, and do it again from scratch. If you can't, you didn't learn it — you watched someone learn it.

**4. Use it to *check*, not to *produce*.** Verification is genuinely safe and genuinely valuable. Producing is where the risk lives. Almost all the good uses in the table above are verification.

**5. Explain it back.** After an explanation, close the window and write it in your own words. That's Level 2 evidence from §0.2. If you can't write it, you don't have it, and the fluent feeling of having understood is misleading — it's the most reliable illusion in this whole business.

**6. Notice dependency.** If you find you can't *start* a problem without opening a chat, that's the symptom. **Action:** one full week with no assistance at all. It'll be slower and uncomfortable, and it will show you exactly what you'd been leaning on.

### The uses that are actually underexploited

Most people either avoid AI entirely or use it to produce. Both are wrong. The high-value uses are:

- **"Grade this derivation."** You derived it. They check. This is closest to what a supervisor does.
- **"What would a reviewer object to in this argument?"** Adversarial pressure on your own work. Directly practises §13.4.
- **"Give me five harder problems on this."** Generating calibrated practice is hard and this is good at it.
- **"I think X because Y. Argue against me."** Pressure-tests your reasoning without giving you conclusions.
- **"Explain equation 4 of this paper"** — *after* you've spent 30 minutes on it yourself.
- **"Here's my research question. What's already been done?"** — as a starting point for a real search, not a substitute (§15.1).

### The scarcity reframe

If your access is rate-limited, that's genuinely a mixed blessing rather than a pure loss.

Unlimited access makes the temptation constant and continuous — there's no moment where you have to decide whether a question is worth it. Limited access forces triage, and triage is the discipline that keeps you learning: you spend your questions on the ones that survived Unstuck steps 1–7, which are exactly the questions worth asking.

**Practically:** batch your questions. Keep a running file of things you're stuck on. Work on them yourself first. Then spend your access on the two or three that survived — with the precise sentence, the minimal reproduction, the printed shapes, and what you already tried (§0.9). That format gets a useful answer in one exchange rather than five.

---

## F.11 Emergency protocols

### The bad day

You've been at it two hours and produced nothing.

1. Stop. Walk for 20 minutes. Outside if possible.
2. Come back and do the **minimum viable day**: 90 minutes of building, 15 of Anki.
3. Pick something easy on purpose. Redo an old exercise. Momentum matters more than progress today.
4. Log it honestly: *"bad day, did the minimum."* That's a completed day.

### The bad week

Nothing has worked for five days.

1. Take a **full day off**. Not a guilty day off — a scheduled one.
2. Come back and run the §F.5 test: is there a real gap, or is this the restart urge?
3. If a real gap: fix that specific thing.
4. If not: **park the current topic** (`PARKED.md`) and move to the next one. Some things only become learnable after you've seen what they're for.
5. Drop to 4 hours a day for the following week. Deliberately.

### "I've been stuck on one thing for two weeks"

You've violated the Unstuck Protocol — step 7 exists exactly for this.

1. Write the precise sentence (step 1).
2. Take the *previous* chapter's checkpoint cold (step 2). Nine times out of ten the blocker is there.
3. If that passes: **park it and move on.** Two weeks is far past the point where continuing pays.
4. Review `PARKED.md` weekly. Watch it resolve itself. It will, and watching that happen is what makes you willing to park things next time.

### "I want to quit"

Do this in order:

1. **Don't decide today.** Never decide to quit on a bad day; the decision is being made by the bad day.
2. **Check the calendar (§F.9).** Is it week 3, month 4, month 9? If so, this is scheduled and it passes.
3. **Read month one of your progress log.** All of it.
4. **Take three days completely off.** Genuinely off.
5. **Then decide**, and if you decide to stop, write down why, honestly. Sometimes stopping is correct — circumstances change, priorities change. A considered decision to stop is respectable. Drift isn't.
6. **If you continue:** restart at the minimum viable day for one week before going back to full hours.

---

## F.12 The five things that matter most

If you remember nothing else from this appendix:

1. **Start before you feel like it.** Ten minutes. Motivation follows action.
2. **On bad days, do the minimum, not nothing.** 90 minutes of building plus Anki is a real day.
3. **Don't restart.** Take the earlier checkpoint cold instead — it takes an hour and answers the question.
4. **Could I do this again tomorrow without help?** The one question that keeps AI assistance from quietly replacing your learning.
5. **The month-four wall is scheduled.** Cut hours, don't stop. It passes.

---

## F.13 One last thing

Everything in this appendix is downstream of a single fact: **you are running a year-long project with no external structure, no deadlines imposed by anyone else, and no one checking.**

That's genuinely hard, and it's hard in a way that has nothing to do with your intelligence or your maths background. The people who complete self-directed technical study aren't the smartest ones. They're the ones who built systems that keep working when motivation doesn't — which is what Chapter 0 and this appendix are.

You've already done the hardest single thing, which is starting properly rather than collecting resources. The rest is showing up, in a structured way, for longer than feels reasonable.

That's it. That's the whole method.

---

*This is the end of the appendices.*
