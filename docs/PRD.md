__PRODUCT REQUIREMENTS DOCUMENT__

__RoboEval__

A Rigorous Evaluation & Failure\-Mode Study of

Open\-Source Robot Learning Policies with Residual RL Fine\-Tuning

__Author__

Rubeno Dechua

__Status__

v1\.0 — Approved

__Last Updated__

May 2026

__Target Role__

Robot Learning Engineer

__One\-Line Pitch__

RoboEval is an open\-source evaluation harness and study that systematically measures where state\-of\-the\-art imitation learning policies break, classifies failure modes, and demonstrates a residual RL loop that recovers the top\-frequency failure — producing a reproducible benchmark, an interactive dashboard, and an arXiv\-style writeup ready for submission\.

# __1\. Executive Summary__

Robot learning is transitioning from laboratory demonstrations to production deployments\. The dominant paradigm — imitation learning via Behavioral Cloning, ACT, and Diffusion Policy — has achieved impressive task success in controlled conditions\. However, the field lacks a rigorous, publicly available study of where these policies systematically fail, how failure modes distribute across policy architectures, and whether lightweight residual RL can recover them\.

RoboEval closes that gap\. The project delivers:

- __An evaluation harness __that benchmarks multiple pretrained policies in simulation with a single config change
- __A robustness suite __that quantifies policy degradation under distribution shift, physical perturbation, and visual noise
- __A failure taxonomy __with six operationally\-defined categories, a classifier, and per\-policy breakdown
- __A residual RL study __showing whether PPO\-based fine\-tuning can recover the highest\-frequency failure mode
- __A public deliverables package __including an interactive dashboard, demo video, MkDocs site, and arXiv\-style writeup

__Why this project for Rubeno specifically__

Rubeno spent 8\+ months at Physical Intelligence evaluating real robot policies, identifying failure modes, and documenting instructional frameworks — this project is a public, reproducible extension of that exact workflow\. It fills the gap between industry experience and open\-source portfolio signal that robot learning hiring teams need to see\.

# __2\. Problem Statement__

## __2\.1  The Gap in the Field__

State\-of\-the\-art robot manipulation policies \(ACT, Diffusion Policy, π0\) are benchmarked primarily on task success rate in nominal conditions\. Three critical gaps remain:

- No standardised public evaluation harness exists for comparing these policies under consistent conditions\.
- Failure modes are qualitatively described in papers but rarely quantified, classified, or cross\-compared across policy architectures\.
- The value of lightweight residual RL on top of pretrained imitation policies is understood in theory but almost no public implementation with measured results exists for newcomers to the field\.

## __2\.2  The Gap for a Candidate__

New\-grad and early\-career robot learning candidates typically demonstrate:

- Toy RL implementations \(CartPole, MuJoCo Ant\) that every CS grad has done
- Course or Kaggle projects with no deployment or reproducibility discipline
- No public evidence of being able to design an experiment, run it cleanly, and communicate the results

RoboEval is designed to fill all three gaps simultaneously and produce artifacts that map directly to responsibilities listed in Robot Learning Engineer and Autonomy Evaluation Engineer job descriptions\.

# __3\. Goals & Non\-Goals__

## __3\.1  Goals__

1. Build and open\-source a reproducible policy evaluation harness runnable on Apple M1 \(8 GB RAM\) with no GPU required for evaluation\.
2. Benchmark Policy A \(`lerobot/act_aloha_sim_transfer_cube_human`\) from the LeRobot model zoo under baseline and perturbed conditions\. Build the harness as policy\-agnostic so additional policies \(e\.g\. a community\-trained Diffusion Policy\) can be added in v1\.1 via a single config flag\.
3. Define, operationalise, and classify at least 6 failure mode categories across ≥150 labelled rollouts\.
4. Train a residual PPO policy on top of the highest\-frequency failure mode and report ΔTSR \(task success rate improvement\) with ablations\.
5. Produce a public demo video, interactive Plotly/Dash dashboard, arXiv\-style PDF writeup, and MkDocs documentation site\.
6. Complete all of the above in 10 weeks on a part\-time schedule \(~15–20 hrs/week\)\.

## __3\.2  Non\-Goals__

The following are explicitly out of scope for v1\.0 to protect timeline:

- Training a full policy from scratch — this requires compute we do not have and adds no differentiation over existing work
- Real\-robot deployment — sim\-only is sufficient and is the standard for eval research
- CUDA / GPU infrastructure — M1 MPS and Colab Pro overflow handle all compute needs
- Novel algorithm contributions — this is an empirical study, not a methods paper
- Comparison to proprietary models \(π0, RT\-2, Helix\) — no public checkpoints available
- __Cross\-policy comparison in v1\.0 __— no sim\-trained Diffusion Policy checkpoint exists publicly for ALOHA Transfer Cube \(see Section 6\.1\)\. v1\.0 is single\-policy \(ACT\); cross\-policy comparison and the multi\-policy perturbation grid are deferred to v1\.1 as a stretch goal contingent on training Diffusion Policy from scratch on Colab or the appearance of a community checkpoint\.
- __Multi\-policy perturbation grid __— for the same reason, the perturbation suite in Section 6\.4 runs on ACT only in v1\.0\. This keeps the 10\-week timeline feasible on a single M1 \(see Section 11\)\.

# __4\. Target Audience__

## __4\.1  Primary: Hiring Teams at Robot Learning Companies__

The project's core external audience is recruiting and engineering teams at companies hiring for Robot Learning Engineer, Autonomy Evaluation Engineer, and ML Research Engineer roles\. Specific targets include:

- Physical Intelligence, Skild AI, 1X Technologies, Figure AI, Dexterity, Covariant — manipulation\-focused
- Boston Dynamics AI Institute, Apptronik, Agility Robotics — humanoid / mobility
- General Motors Autonomy, Waymo, Cruise Automation — autonomy evaluation adjacent

## __4\.2  Secondary: The Robot Learning Research Community__

Open\-sourcing the harness and study benefits researchers who want a plug\-and\-play evaluation baseline\. GitHub stars, forks, citations, and community issues all add long\-term professional signal\.

## __4\.3  Tertiary: Rubeno Himself__

The project functions as structured upskilling\. Every phase introduces a concept that will come up in technical interviews: reward design, RL training loops, statistical experimental design, and research communication\.

# __5\. Technical Architecture__

## __5\.1  System Overview__

RoboEval is structured as a Python library with four major components: Environment Layer, Policy Layer, Evaluation Engine, and Analysis & Reporting Layer\. These are orchestrated by Hydra configuration files and logged to Weights & Biases\.

__Architecture Principle__

Every experiment must be fully reproducible from a single YAML config file\. Given a config, the eval harness produces identical results \(modulo fixed random seed\) with one command: python evaluate\.py config=baseline/act\_nominal\.yaml

## __5\.2  Repository Structure__

roboeval/
├── configs/              \# Hydra YAML configs per experiment
│   ├── baseline/         \# Nominal eval configs \(one per policy\)
│   ├── perturbation/     \# Robustness suite configs
│   └── residual\_rl/      \# PPO fine\-tuning configs
├── roboeval/             \# Core library \(typed Python\)
│   ├── envs/             \# Environment wrappers \(Gymnasium API\)
│   ├── policies/         \# Policy loader \+ inference wrappers
│   ├── evaluation/       \# Rollout engine, metric collectors
│   ├── taxonomy/         \# Failure mode classifier
│   └── residual/         \# Residual RL trainer \(SB3 PPO\)
├── analysis/             \# Notebooks \+ Plotly dashboard
├── docs/                 \# MkDocs source
├── tests/                \# pytest unit \+ integration tests
├── \.github/workflows/    \# CI: lint, type\-check, smoke\-test
├── pyproject\.toml        \# Ruff, mypy, dependency config
└── README\.md

## __5\.3  Technology Stack__

__Category__

__Tool / Library__

__Purpose__

__M1 Compatible__

Simulation

MuJoCo 3 \+ gym\-aloha

Primary manipulation env

✓ CPU/MPS

Simulation

PyBullet \(fallback\)

Lightweight alternative env

✓ CPU

Policies

LeRobot \(HuggingFace\)

ACT & Diffusion Policy checkpoints

✓ MPS inference

RL Training

Stable\-Baselines3

PPO residual policy training

✓ MPS

RL Training

Gymnasium

Environment wrapper API

✓

Config Mgmt

Hydra

Reproducible experiment configs

✓

Logging

Weights & Biases \(free tier\)

Run tracking, plots, artifacts

✓ \(cloud\)

Visualization

Plotly \+ Dash

Interactive eval dashboard

✓

Viz \(static\)

Seaborn \+ Matplotlib

Publication\-quality plots

✓

Code Quality

Ruff \+ mypy \+ pre\-commit

Linting, typing, git hooks

✓

CI/CD

GitHub Actions

Auto\-test on push

✓ \(cloud\)

Docs

MkDocs \+ mkdocstrings

Auto\-generated API docs site

✓

GPU Overflow

Google Colab Pro \(~$10/mo\)

Larger inference, RL training

N/A — cloud

# __6\. Evaluation Design__

## __6\.1  Policies Under Study__

v1\.0 benchmarks a single pretrained policy, selected for public availability of a sim\-trained checkpoint compatible with the MuJoCo gym\-aloha Transfer Cube task:

- __Policy A — ACT \(Action Chunking with Transformers\): __HuggingFace ID `lerobot/act_aloha_sim_transfer_cube_human`\. Transformer\-based; outputs action chunks; reports ~83% task success rate at 80k training steps on the source task\. Verified to exist on the HuggingFace Hub \(May 2026\)\.

Diffusion Policy was originally planned as Policies B and C\. A May 2026 audit of the public `lerobot/*` HuggingFace organisation confirmed that __no sim\-trained Diffusion Policy checkpoint exists for ALOHA Transfer Cube__: only `lerobot/diffusion_pusht` and `lerobot/diffusion_pusht_keypoints` are published, both for the PushT task\. Training Diffusion Policy from scratch on the `lerobot/aloha_sim_transfer_cube_human` dataset is out of scope for v1\.0 \(see Section 3\.2\) and is deferred to v1\.1 as a stretch goal\.

If the ACT checkpoint proves incompatible with M1 MPS inference, a CPU\-only fallback will substitute\. Colab Pro is reserved for inference runs exceeding 4 GB VRAM\.

## __6\.2  Evaluation Tasks__

All evaluations use the gym\-aloha manipulation environment \(part of the LeRobot ecosystem\), which runs on CPU without CUDA:

- __Transfer Cube Task: __Move a cube from one receptacle to another — standard, well\-understood baseline task
- __Insertion Task \(stretch goal\): __Insert a peg into a socket — higher precision requirement, exposes more failure modes

__Task success criterion \(Transfer Cube\):__ a rollout is considered successful when the cube's centre\-of\-mass z\-position exceeds `z_success = 0.05 m` __and__ its xy\-position lies inside the target receptacle bounding box \(default ±0\.05 m around the receptacle centre\) for `N_dwell = 5` consecutive simulation steps\. The defaults `(z_success, xy_tolerance, N_dwell) = (0.05 m, 0.05 m, 5)` are Week 1 placeholders and must be tuned against the `lerobot/act_aloha_sim_transfer_cube_human` checkpoint during the smoke\-test phase so that nominal\-condition TSR roughly matches the ~83% figure reported on the model card\. The tuned values are then frozen for all subsequent experiments\.

## __6\.3  Metrics__

__Metric__

__Definition__

__Tool / Source__

__Target__

Task Success Rate \(TSR\)

% rollouts completing task end\-to\-end

Custom eval loop

Reported ± std

Time\-to\-Success \(TTS\)

Median steps to task completion

Rollout logs

Reported

Perturbation Recovery Rate

TSR after mid\-rollout object shift

Eval harness

> baseline TSR × 0\.5

Failure Mode Distribution

% rollouts per failure category

Taxonomy classifier

≥5 categories

Residual RL Delta \(ΔTSR\)

TSR improvement over frozen base policy

Ablation table

> \+10% on target failure

Eval Reproducibility \(σ\)

Std dev across 3 random seeds per config

Eval harness

σ < 5%

*All metrics are reported as mean ± standard deviation across 3 random seeds and ≥50 rollouts per condition\. This is the minimum bar for results to be credible in a research context\.*

## __6\.4  Robustness Perturbation Suite__

The perturbation suite stresses __Policy A \(ACT\) only__ in v1\.0 \(see Section 3\.2 for the rationale and v1\.1 expansion plan\) along four axes\. Each axis defines a range of perturbation intensities and reports TSR as a function of intensity — producing degradation curves rather than single\-point measurements:

- __Spatial perturbation: __Object start position shifted ±1cm, ±3cm, ±5cm from nominal
- __Visual perturbation: __Lighting intensity varied ±30%, ±60%; distractor object added to scene
- __Dynamic perturbation: __Object pushed 2cm mid\-rollout at step 25%, 50%, 75% of nominal completion
- __Temporal perturbation: __Action execution delayed by 1, 3, 5 steps \(simulates real\-world latency\)

# __7\. Failure Mode Taxonomy__

## __7\.1  Design Principles__

The taxonomy is designed to be operationally unambiguous: any two labellers should agree on the failure category for a given rollout\. Each category has a precise definition, a concrete example, and a measurable detection rule\.

## __7\.2  Taxonomy__

__Category__

__Definition__

__Example__

Grasp Failure

Robot contacts object but drops or misses

Finger collision causes object to slide

Approach Failure

Robot reaches wrong position/orientation before contact

End\-effector overshoots target by >5cm

Recovery Failure

Policy cannot correct after perturbation

Object shifted 3cm — policy ignores and continues old trajectory

Action Oscillation

Policy outputs rapidly alternating contradictory actions

End\-effector jitters in place for >5 steps

Timeout

Task not completed within step budget \(default __400 steps__ for Transfer Cube; tunable per task in Hydra config\)

Policy plateaus with no progress for 50\+ steps

Visual Confusion

Policy error correlates with visual change \(lighting, distractor\)

Success drops >30% under changed lighting

## __7\.3  Labelling Protocol__

1. Run evaluation harness; save full rollout trajectory \(observations, actions, rewards\) per episode
2. Auto\-classify rollouts into failure categories using a rule\-based classifier \(threshold on position error, action variance, step count\)
3. Manually review a stratified sample of 30 rollouts \(5 per category\) to validate classifier accuracy
4. Report inter\-rater reliability on the sample \(target: agreement > 85%\)
5. Produce per\-policy failure distribution heatmap

# __8\. Residual RL Design__

## __8\.1  Rationale__

Residual RL trains a small correction policy on top of a frozen base policy\. Rather than replacing the base policy, it learns to predict the delta between what the base policy does and what should be done in failure\-prone states\. This is much cheaper to train than a policy from scratch and maps directly to real\-world workflows where pretrained policies are valuable but imperfect\.

__Why Residual RL and not Full RL from scratch?__

Full RL from scratch on manipulation tasks requires millions of environment steps and significant GPU time — not feasible on M1 8 GB\. Residual RL operates in a narrow region of the action space near the base policy, converges faster, and is the approach actually used by robot learning teams to iterate on deployed policies\. This choice demonstrates research taste, not hardware limitation\.

## __8\.2  Implementation__

- __Base policy: __Frozen ACT checkpoint \(Policy A\) — highest failure rate in the most targeted failure category
- __Residual policy: __Small MLP \(2 hidden layers, 256 units\) trained with PPO via Stable\-Baselines3
- __Reward signal: __Sparse success reward \+ shaped distance\-to\-goal term \(ablated separately\)
- __Training budget: __500k environment steps \(feasible in ~4 hours on M1 CPU / 1 hour on Colab\)
- __Action composition: __Final action = base\_action \+ α × residual\_action, where α is a learnable scalar

## __8\.3  Ablation Plan__

Three conditions are reported to make the study credible:

- __Condition A — Frozen base only: __ACT policy, no residual \(baseline\)
- __Condition B — Residual RL \(sparse reward\): __PPO residual with binary success signal
- __Condition C — Residual RL \(shaped reward\): __PPO residual with distance\-to\-goal shaping

*ΔTSR = TSR\(Condition B or C\) − TSR\(Condition A\)\. A positive ΔTSR > 10% on the target failure mode is the success criterion\. A negative or null result is still reported with analysis — honest null results are valued in research\.*

# __9\. Deliverables__

__Deliverable__

__Description__

__Audience Signal__

GitHub Repository

Typed Python codebase, CI, full README, MIT license, one\-command eval

Robot learning / MLOps teams

Evaluation Harness

Pluggable harness: swap policy or env with single config change

Infra / research engineers

Benchmark Report \(PDF\)

arXiv\-style writeup: method, results, ablations, limitations

Research\-leaning hiring managers

Interactive Dashboard

Plotly/Dash web app: filter by policy, metric, failure mode

Product / applied ML teams

Demo Video \(90s\)

Screen\-recorded rollouts showing key failure modes \+ recovery delta

All recruiters — top of funnel

MkDocs Site

Auto\-generated API docs, hosted on GitHub Pages

Engineering due\-diligence

W&B Report

Public Weights & Biases run dashboard with all metrics

ML practitioners

## __9\.1  Industry\-Standard Quality Checklist__

Every deliverable must pass the following before public release:

- GitHub repository: zero lint errors \(Ruff\), zero mypy type errors, CI passing on push, >70% test coverage, README with one\-command eval instructions
- Writeup: methods section reproducible from description alone; results section includes uncertainty estimates; limitations section present and honest
- Dashboard: loads in <3 seconds; mobile\-responsive; all plots have axis labels, units, and titles
- Demo video: narrated; failure modes visually annotated; ≤90 seconds; 1080p minimum
- MkDocs site: all public functions documented; quickstart example runs without modification

# __10\. Project Plan__

## __10\.1  Phase Overview__

__Phase__

__Name__

__Key Deliverables__

__Duration__

1

__Foundation & Environment Setup__

Repo scaffold, env verified, CI passing

Week 1

2

__Baseline Policy Evaluation__

Baseline metrics on Policy A \(ACT\); policy\-agnostic harness v1

Weeks 2–3

3

__Robustness & Failure Mode Study__

Perturbation suite, failure taxonomy, ablation plots

Weeks 4–5

4

__Residual RL Fine\-tuning__

Trained residual policy, delta metrics, ablation

Weeks 6–7

5

__Dashboard & Writeup__

Plotly dashboard, blog post draft, demo video

Week 8

6

__Polish & Launch__

Peer review, README, arXiv\-style PDF, public release

Weeks 9–10

## __10\.2  Week\-by\-Week Schedule__

Based on 15–20 focused hours per week, executable alongside part\-time or full\-time work:

__Wk__

__Phase__

__Tasks__

__Done When…__

1

Setup

Repo scaffold, CI, env install, smoke\-test policy rollout

First rollout renders without crash

2

Eval v1

Baseline TSR on Policy A \(ACT\), 50 rollouts, log to W&B

W&B dashboard showing TSR ± std

3

Eval v1

Generalise harness to policy\-agnostic loader \(ready for v1\.1 DP\); expand ACT baseline to 3 seeds × 50 rollouts × nominal conditions

Harness loads any LeRobot policy via single config flag; ACT baseline TSR ± std logged to W&B

4

Robustness

Perturbation suite \(ACT only\): object shift, lighting, distractor, action delay

Perturbation TSR table complete for Policy A

5

Failure Study

Failure taxonomy: label 150\+ rollouts, build classifier, plot distribution

Taxonomy heatmap \+ per\-policy breakdown

6

Residual RL

PPO residual policy setup; train on top\-1 failure mode in sim

Training curve visible in W&B

7

Residual RL

Ablation: residual vs\. frozen baseline; compute ΔTSR

Ablation table with ≥3 conditions

8

Comms

Plotly dashboard; 90\-second demo video; blog post draft

Dashboard deployed; video uploaded

9

Polish

arXiv\-style PDF; MkDocs site; peer review \(1 external reader\)

PDF readable start\-to\-finish

10

Launch

Tag v1\.0 release; post on LinkedIn \+ r/MachineLearning; cold outreach

5\+ GitHub stars within first week

## __10\.3  Definition of Done__

The project is done when all of the following are true simultaneously:

- GitHub repo is public, CI is green, and README produces a working eval run on a fresh clone
- All 6 failure categories have ≥20 labelled rollout examples
- Residual RL ablation table shows results for all 3 conditions with ≥3 seeds
- Dashboard is deployed and accessible via public URL
- Blog post is published \(GitHub Pages, Medium, or personal site\)
- Demo video is posted on YouTube or LinkedIn

# __11\. Risks & Mitigations__

__Risk__

__Likelihood__

__Impact__

__Mitigation__

Real\-trained checkpoint loaded into sim env

Medium

Near\-zero baseline TSR; eval results uninterpretable

Only use sim\-trained variants from the LeRobot zoo \(e\.g\. `lerobot/act_aloha_sim_transfer_cube_human`\); verify on Day 1 by running the nominal\-condition rollout and confirming TSR > 50% before any further work

M1 RAM OOM during inference

High

Blocks eval loop

Use smaller ACT checkpoint; reduce batch size to 1; fallback to PyBullet env

LeRobot API breaking changes

Medium

Eval harness breaks

Pin dependency versions; use Docker for reproducibility

Residual RL doesn't improve TSR

Medium

Weak project story

Reframe as negative result with analysis — still publishable; focus writeup on failure taxonomy instead

Eval is too slow on CPU

Medium

Limits rollout count

Vectorize env with Gymnasium AsyncVectorEnv; parallelise across CPU cores

Project scope creep

High

Misses 10\-week target

Lock MVP scope at end of Phase 2; additional features go to backlog

# __12\. Success Criteria__

## __12\.1  Project Success \(Technical\)__

- Evaluation harness runs Policy A \(ACT\) end\-to\-end without manual intervention, and is policy\-agnostic enough to accept a second LeRobot policy via config change with no code changes
- At least 150 rollouts classified into failure taxonomy with >85% inter\-rater agreement
- Residual RL ablation complete with all 3 conditions and error bars
- All 7 deliverables shipped to public\-facing channels

## __12\.2  Career Success \(Hiring Signal\)__

- GitHub repository receives ≥5 stars within 2 weeks of launch
- At least 1 cold outreach to a robot learning company receives a positive response citing the project
- Project is discussed substantively in at least 1 technical interview
- At least 1 hiring manager or researcher comments on or shares the writeup / dashboard

## __12\.3  Personal Learning Success__

- Able to explain reward design tradeoffs \(sparse vs\. shaped vs\. learned\) from first principles
- Able to walk through a residual RL training loop in a whiteboard interview without notes
- Familiar with at least 3 papers per topic area covered in the writeup's related work section

# __13\. Interview Question Alignment__

This table maps the most common robot learning interview questions to specific project artifacts, so responses are grounded in real, demonstrated work rather than hypotheticals:

__Common Interview Question__

__Project Artifact__

__Your Answer Angle__

Walk me through a time you evaluated a model rigorously

Eval harness \+ metrics table

Built reproducible harness; ran 50 rollouts per config; reported mean ± std across 3 seeds

How do you design a reward function?

Reward ablation in Phase 3

Compared sparse vs\. dense vs\. VLM\-as\-judge; showed tradeoffs in sample efficiency and reward hacking

What does sim\-to\-real gap mean to you?

Phase 3 perturbation study

Systematically varied physics/visual params in sim; quantified TSR degradation curves

Have you worked with RL on real systems?

PI experience \+ residual RL

Evaluated physical robot policies at PI; project extends that to RL fine\-tuning in sim

Tell me about a failure and what you learned

Failure taxonomy section

Classified 6 failure modes; recovery rate metric; residual RL targeted the highest\-frequency failure

How do you communicate technical findings?

Dashboard \+ blog post \+ video

Built interactive Plotly dashboard; wrote arXiv\-style report; produced 90\-second demo video

# __14\. Getting Started — Day 1 Checklist__

Complete these steps before writing any project code:

1. Create GitHub repository: roboeval — set to public, MIT license, Python \.gitignore
2. Install uv \(fast Python package manager\): curl \-LsSf https://astral\.sh/uv/install\.sh | sh
3. Create project environment: uv venv \.venv && source \.venv/bin/activate
4. Install core dependencies \(pin LeRobot to a known\-good version\): uv add 'lerobot==0\.4\.4' gymnasium mujoco stable\-baselines3 hydra\-core wandb plotly ruff mypy
5. Verify M1 MPS availability: python \-c "import torch; print\(torch\.backends\.mps\.is\_available\(\)\)" — should print True
6. Run LeRobot smoke test: python \-c "from lerobot\.policies\.act\.modeling\_act import ACTPolicy; print\('ACT loaded'\)" — note that LeRobot 0\.4\.x removed the `lerobot\.common` namespace; import policies directly from `lerobot\.policies\.*`
7. Create first W&B project at wandb\.ai — free account, project name: roboeval
8. Set up pre\-commit hooks: uv add \-\-dev pre\-commit && pre\-commit install
9. Read these 3 papers before writing any evaluation code: ACT \(Zhao et al\. 2023\), Diffusion Policy \(Chi et al\. 2023\), ResiP \(residual RL for manipulation, 2022\)
10. Block calendar: 3\-hour focused sessions on at least 5 days per week for 10 weeks

__The Most Important Instruction in This Document__

The gap between a good project and a great one is the writeup\. Code without communication is invisible\. Every week, write one paragraph describing what you did, what surprised you, and what you plan next\. The blog post writes itself from these notes\. Do not leave writing until Week 8\.

# __Appendix: Reference Papers__

These are the minimum reading list before and during the project\. Reading these papers is what allows the writeup to situate itself credibly in the field:

- __ACT — __Zhao et al\. \(2023\)\. Learning Fine\-Grained Bimanual Manipulation with Low\-Cost Hardware\. RSS 2023\.
- __Diffusion Policy — __Chi et al\. \(2023\)\. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion\. RSS 2023\.
- __LeRobot — __Cadene et al\. \(2024\)\. LeRobot: State\-of\-the\-art Machine Learning for Real\-World Robotics\. GitHub\.
- __Residual RL — __Silver et al\. \(2019\)\. Residual Policy Learning\. arXiv:1812\.06298\.
- __IWR / Interactive Learning — __Mandlekar et al\. \(2020\)\. IRIS: Implicit Reinforcement without Interaction\. ICRA 2020\.
- __Robomimic — __Mandlekar et al\. \(2021\)\. What Matters in Learning from Offline Human Demonstrations for Robot Manipulation\. CoRL 2021\.
- __VLA Survey — __Firoozi et al\. \(2023\)\. Foundation Models in Robotics: Applications, Challenges, and the Future\. arXiv:2312\.07843\.

Document prepared by Rubeno Dechua — May 2026\. RoboEval is an independent portfolio project\. All tools referenced are open\-source or free\-tier\.

 