# Mechanistic interpretability

## Motivation

**What I'm interested in.** I want to understand how neural networks actually work: what algorithms they learn and where those live in the weights and activations. It's the kind of mechanism-level question that shows up in physics when you care about explaining behaviour from first principles instead of treating the system as a black box.

**Vision.** I believe there is a fundamental, general theory that can explain neural networks, and that we could use it to *build* networks instead of only training on data without really knowing how learning happens. In physics, we already have classical and quantum mechanics: they tell us how physical systems behave, and we design and build systems on the basis of those laws. I'm excited by the possibility of something analogous for neural networks.

**How my past research connects.** I've done fundamental research on heat transfer and phonons in atomic systems. In condensed matter, you're trying to understand patterns, collective behaviour, and how systems respond to perturbations. I see a strong parallel with mechanistic interpretability: same desire to open the black box and understand what's going on underneath. That's the kind of understanding I think we need if we want systems that are reliable, safe, and aligned.

For some of my past research, see my [Google Scholar](https://scholar.google.com/citations?user=HJJ7NTUAAAAJ&hl=en).

## Projects

With that motivation in mind, I've started exploring the current state of research in mechanistic interpretability and sharing some of these initial explorations here. The first is [logit lens and activation patching](logit_lens_patching/logit_lens_and_patching.ipynb): when and where a model's prediction forms, and which layers matter for it. That's the kind of mechanism-level view I'm after.

### logit_lens_patching

**Logit lens & activation patching** on a language model (TransformerLens + GPT-2 Small).

- [README](logit_lens_patching/README.md)
- [Notebook](logit_lens_patching/logit_lens_and_patching.ipynb)
