

The fundamental problem of sequence **modeling is compressing context into a smaller state**. The efficiency vs effectiveness trade-off of sequence models is characterized by how well they compress their state. Efficient models have small state, effective models have a state that contains all the necessary context information. Until this paper, the top effective model was transformer, while the top efficient model was S4.

All the speed-up and memory savings of SSMs/S4 vs. transformers came at the cost of accuracy. The key weakness is that SSMs are not able to perform content-based reasoning (focus/ignore particular inputs). A fundamental principle for building sequence models is selectivity, or the context-aware ability to focus on/filter out inputs into a sequential state. A selection mechanism should control how information propagates or interacts along the sequence dimension. One method of incorporating a selection mechanism into models is by letting their parameters that affect interactions along the sequence be input-dependent. Since the matrixes describing the model $(A, B, C, \Delta)$ are fixed, SSMs treat each token equally. In contrast to the static nature of SSMs, transformers are very dynamic: they re-compute attention at each step, resulting in being input dependent/aware. This paper aims at relaxing the static-ness of S4, making it dependent on the input. 

The main contributions of this paper is selective SSM, composed of:
- **selective scan algorithm**, which allows the model to filter (ir)relevant information
- (*) **hardware-aware adaptation** that allows for efficient storage of intermediate results through parallel scan, kernel fusion (of discretization step, selective scan algorithm, and multiplication with C), and recomputation (collection of optimizations that come from the *flashattention* paper)

Selective SSM is wrapped into an architecture (H3 with gate mechanism) to form a **Mamba block** (to be used like an attention block within a network). 

In short, the **main difference between Mamba (S6) and its predecessor S4, is simply making several parameters functions of the input**. Mamba allows the transition through the hidden space to be dependent on the current input (not on the previous hidden states though, $A$ is still fixed).

<br>

---

(*) Why the hardware-aware adaptation? While $A$ is still a parameter, everything else $(B, C, \Delta)$ is computed from the input. In doing so, everything gains a dimension. To make it efficient inside the GPU, the paper proposes some flashattention-alike tricks to make it still fast. 

----

<br>

![image](https://github.com/paulinamoskwa/ml-papers/assets/104844027/1d2e0005-34f8-4f7e-a320-282f8945df90)
