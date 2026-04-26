# The Long Walk to an Incident Commander

*How a stray clip about the LA fires turned into my first reinforcement learning project.*

---

## The empty shortlist

When the OpenEnv hackathon was announced, I did not have an idea.

I had a shortlist, sure, the kind every hobbyist machine-learning person carries around in the back of their head. A code-review agent. A structured-output LLM judge. Something to do with compliance reports, an idea I abandoned within a day because the moment I tried to write a one-paragraph pitch I started yawning. There is a useful diagnostic in that, I think. If your own pitch bores you in draft, it will bore everybody else louder.

So I scrolled. Embarrassingly enough, that is where the project really starts.

It was around one in the morning, and I was doom-scrolling YouTube shorts in the kind of fugue where individual videos stop registering and become a slurry of noise. An NBC Bay Area clip drifted past. Then aerial footage of the LA fires, hills the colour of rust, voice-overs droning numbers about acres and containment percentages. What I remember is the way one reporter said, almost in passing, *"the incident commander has decided to pull crews back"*. A single human being, deciding at three in the morning their time, where to put the people, where to drop the retardant, which neighbourhoods to write off. I closed the app, went to bed, and didn't think about it again for a couple of days.

But it kept coming back. Quietly, at first, while I was supposed to be reading the hackathon brief. The themes mapped almost too cleanly onto the shape of the thing that incident commander on the clip was doing: incomplete information, hard resource limits, weather you cannot argue with, civilians you cannot disappoint. I scribbled *"Wildfire Incident Commander as an RL task"* on a sticky note and pressed it to the side of my monitor. The note is still there, slightly buckled at the corner now, with coffee on it.

There was one problem with this plan. I had never actually trained a reinforcement learning policy.

---

## How I got into RL in the first place

My exposure to RL up to this point was almost entirely cinematic.

I had grown up on those grainy DeepMind Atari videos, the ones where a tiny green paddle slowly figures out it can tunnel through the side of a Breakout wall and bounce the ball around behind it. I remember rewatching that clip on loop the first time I saw it and feeling something genuinely uncanny. The agent had not been told about the tunnel. Nobody coded the tunnel. It just appeared, somewhere in the loss landscape, as the cheapest way to keep the ball alive.

From there it was the usual pipeline. The *AlphaGo* documentary, late one weekend. OpenAI's hide-and-seek video where the agents start surfing on boxes their opponents are still trying to lock down. Two Minute Papers explaining AlphaStar and OpenAI Five with that delighted Hungarian cadence. I read Sutton and Barto in the way most people read Sutton and Barto, which is to say, three chapters in great detail and the rest in spirit. I read the Mnih DQN paper, the Schulman PPO paper, eventually the DeepSeek-R1 work and the GRPO derivations, and I poked at a couple of CartPole notebooks. But I had never actually trained a policy that mattered. RL had this folkloric reputation around it, finicky, expensive, vibes-based, the part of the deep-learning toolbox most likely to silently fail in interesting ways. I had a healthy fear of the field.

---

## The confession

So the confession I should make early is that this was my first real reinforcement learning project. OpenEnv was even newer to me. I came in cold.

What kept me from bailing in the first three days was, paradoxically, exactly that newness. I was already accepting I would be uncomfortable. I figured I might as well be uncomfortable about something that genuinely interested me. The Dunning-Kruger trough was waiting for me regardless. Better to fall into it doing something I would be proud to talk about afterwards.

So I started reading.

---

## The Rothermel rabbit hole

For the unfamiliar: Richard Rothermel published the canonical surface-fire spread model in 1972, in a US Forest Service technical report titled, with monastic plainness, *"A Mathematical Model for Predicting Fire Spread in Wildland Fuels"*. It is roughly thirty pages, and it underwrites almost every operational fire-behaviour predictor used in the field since. BehavePlus, FARSITE, FlamMap, different tools, different abstractions, the same skeleton. I downloaded the PDF at four in the afternoon and was still reading it at midnight. I was unprepared for how much *taste* there was in those equations, the way Rothermel had to balance theoretical fidelity against parameters a real-world ranger could plausibly measure with a pole and a moisture probe. There is a particular kind of engineering elegance in a model that survives that long under field conditions.

What surprised me more, and what tipped this project from "interesting" to "I have to do this", was how thin the work at the *intersection* of language models, reinforcement learning, and wildfire response was. There are RL papers on fire suppression (Subramanian and Crowley's forest-fire DRL work, Julian and Kochenderfer on aircraft routing for wildfire surveillance, the FireCommander multi-UAV environment). There are LLM-as-agent papers on disaster response. There is an entire operations-research literature on resource allocation in incident command. But the middle of that Venn diagram, where an LLM is the actor inside an RL loop on a Rothermel-style spread environment, was almost empty. For a solo entrant that is a gift. Judges will see ten polished agentic-coding projects for every one project that even *attempts* something this far off the beaten path. I would rather be the rough sketch of something unusual than the tenth-best version of something normal.

---

## The metamorphosis

My first design was nothing like what I ended up shipping.

The initial prototype was a single-step decision agent. The model would receive a snapshot of the fire (a textual map, a brief weather summary, a list of crews) and produce *one* action. The environment would run a thirty-step simulation under a fixed policy, then report back a number. I would train against that number. Clean, simple, tractable. I built it out over two evenings.

It was bad. Not in the "the loss is high" way, but in the "this isn't actually the problem I want to solve" way. A single decision under a thirty-step rollout teaches the model almost nothing about *sequencing*. It teaches it to be a one-shot triagist, a useful skill in narrow contexts, but not what an incident commander actually does. An IC's job is to keep deciding as the situation degrades, to revise, to recover when a crew gets hurt and the wind shifts and the second ignition happens behind them. None of that was in v1.

The metamorphosis happened gradually. I added a step penalty so the model could not loiter. I added a terminal reward for population saved, and immediately the gradient washed out because the terminal reward was rare and small relative to per-step noise. I scaled the terminal reward up; the model learnt to game the per-step component instead. I added a *briefing*, a written paragraph describing priority zones, infrastructure, and the wind forecast, partly because real ICs read briefings, and partly because it gave me an honest measurement of instruction following. I added a curriculum because the hard tier, on its own, produced a flat reward curve and a sad-looking W&B chart. I added a second reward function for JSON validity because GRPO begs for it. I added a heuristic continuation step because I was burning GPU minutes making the model generate seven extra times per prompt, when the gradient only flowed through the *first* generation anyway.

Each of those decisions came from running the thing and watching it fail in a specific, legible way. The system I ended up with is not the system I designed. It is the system that survived.

---

## What I want the judges to take away

There were nights, there are always nights, where the project felt absurd. A solo participant. First RL project. A custom environment. A custom reward decomposition. SFT, then GRPO, on a 7B model, on Colab, with a curriculum controller and a callback that rebuilds the dataset mid-run. I made all the canonical mistakes, including, briefly, training on a dataset whose seeds did not match the seeds the reward function rolled out against, which is the GRPO equivalent of grading a student's geography exam by asking them about a different country. I found it the way you always find these things, by squinting at a sample completion at 2 a.m. and thinking *wait, that does not match*.

But I learned more in three weeks than I would have from a semester of well-mannered tutorials. Reinforcement learning, as a sub-field, is unforgiving in a productive way. It does not let you confuse "the loss is going down" with "the model is doing the thing". You have to look. You have to read rollouts. You have to talk to your model the way an IC talks to a crew chief, patiently, specifically, willing to be told the plan is wrong.

If a judge reads this far, I want to be candid about what I think the contribution actually is. The trained model does not beat the heuristic on hard tier. It approaches it on medium (+5.74 vs. +6.31) and falls short on hard, and I would rather submit honest numbers than goose them. The headline is not the leaderboard, it is the *artifact*: a typed, OpenEnv-compliant environment with a Rothermel-flavoured spread model, a decomposed reward built for GRPO, a serialiser that keeps prompt length sub-linear in grid size, a parser that refuses to crash on malformed completions, and an end-to-end training recipe somebody else can pick up tomorrow and improve on. Plus a frank post-mortem of every bug I hit along the way, which is, I suspect, more useful to the next person trying this than another tenth of a reward point would be.

I started this project because of a one-minute clip about somebody else's bad night. I am finishing it knowing more about reinforcement learning, more about wildland fire science, and a little more about my own tolerance for ambiguity than I did three weeks ago.

The sticky note is staying on the monitor.

---

## The five bugs that taught me the most

A short post-mortem, because the bugs are the part of the story you actually learn from.

1. **Frozen dataset, live curriculum.** The controller promoted to *medium* at step 10 and *hard* at step 20, but the prompt dataset was built once before training and never refreshed. The model was happily being scored on easy prompts while the dashboard insisted it was on hard. Fixed with a `TrainerCallback` that rebuilds the dataset on tier change.
2. **Truncated rollouts never saw terminal reward.** v1 ran a fixed 15 step rollout per completion. Hard tier needs at least 80 steps before the +5 survival bonus can fire, so GRPO was optimising against per-step deltas only. v2 runs to `env.done`. Twice as slow, gradient signal night-and-day better.
3. **Prompt and reward state mismatch.** Each dataset row was generated with a fresh random seed, and the reward function picked *another* fresh seed at scoring time. The model was being graded on a different fire than the one in its prompt. Now every row carries its `seed`, and the reward function resets to that exact `(tier, seed)`.
4. **Wasted inner generations.** v1 called `model.generate()` seven extra times per completion to build a multi-step rollout, but GRPO gradients only flow through the originally sampled completion. Those seven calls were expensive noise. Cutting `MODEL_STEPS` to 1 and letting the heuristic finish the episode dropped wall-clock per step by about 70%.
5. **Format reward crashing on `obs=None`.** The action parser reads `obs.grid` to validate spatial fields, so calling it for a pure JSON-validity check crashed. A standalone `check_json_format()` that does not need an obs solved it in twenty lines.

---

## Links

- Live environment on Hugging Face: [`Eshit/Wildfire-Containment-Simulator`](https://huggingface.co/spaces/Eshit/Wildfire-Containment-Simulator)
- Source: [`Abrodolph/Wildfire-Containment-Simulator`](https://github.com/Abrodolph/Wildfire-Containment-Simulator)
- Trained model: [`Eshit/wildfire-grpo-7b`](https://huggingface.co/Eshit/wildfire-grpo-7b)
- W&B run: [`saini-eshit-/wildfire-grpo/runs/dnz56kuu`](https://wandb.ai/saini-eshit-/wildfire-grpo/runs/dnz56kuu)
- GRPO notebook: [`training/grpo_v2_colab.ipynb`](training/grpo_v2_colab.ipynb)
- SFT notebook: [`training/sft_colab.ipynb`](training/sft_colab.ipynb)
- Top-level overview: [`README.md`](README.md)

*Eshit, April 2026*
