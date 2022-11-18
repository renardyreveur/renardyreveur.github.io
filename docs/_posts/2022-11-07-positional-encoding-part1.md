---
title:  "Positional Encodings of Attention Mechanisms: Background (Part 1)"
categories:
  - Blog
tags:
  - attention
  - transformers
  - deep-learning 
  - positional-encodings

excerpt: How do you design a model such that, despite the limited context of weighted sums, manages to capture the key properties of the given data?

---

Check out [part 2](https://renardyreveur.github.io/blog/positional-encoding-part2)!

## Introduction

> All the world's a stage, \\
> And all the men and women merely Players; \\
> They have their exits and their entrances...

Ever have that feeling when you stop and simply *perceive* the world around you, then become awe-struck at its beautiful orchestration?
I certainly do, and amidst the frustration in attempting to describe that exact impression, you somehow find wisdom in Shakespeare's words.

The monologue in *'As you like it'* presents us with a perspective we can take advantage of. If the world is a stage and everything in it are actors, the feelings must come from two things: the mise-en-scène of the stage and, more prominently, the **interactions** of the players within that scene.

In mathematical language, the former would refer to the constraints, definitions, and assumptions we make for the model. The interactions would be composed of the following:

- Interactions between different actors or with oneself, denoted by the multiplication of variables(or values derived from variables)
- The relative strength between interactions given as ratios/weights/coefficients (that are multiplied to the variables)
- Scaled/gated/convolved resultant values of interactions through function compositions

So at the risk of over-simplification, the language of description is a series of **well-structured multiplications and additions**. This framework helps abstract the dynamic and complex nature of our view of the stage. Yet, without a strict understanding of specific properties that we'd like the model to respect, the information enclosed within the context of weighted sums can become ambiguous.

**How do you infer distinct meanings from each of the interactions represented with additions and multiplications?**
{: .notice--primary} 

Maybe an opinionated mise-en-scene can enforce a specific rule; perhaps the actors can carry a particular identifier by wearing a tag on their arm, and so on. There isn't one definite answer to this, but many. In this post(especially in part 2!), we'll look at **how various authors injected specificity in a sequence's order** as they tried to direct a play called sequential phenomena(natural language).

{% include figure image_path="/assets/positional_encoding/order.png" alt="a line of block people"%}


## Neural Networks

Neural networks are a family of models that are widely used to represent higher-order tasks and problems. The premise is that we define an event we want to describe with the **data it generates**. Then, by iteratively fitting the data to a pre-defined function, we hope that the fully converged model produces a similar distribution to the trained data. In doing so, two things need to happen. First, the *model structure* needs to be defined, and then the *model weights and biases must be updated* via gradient descent against an objective function. 

The model is made up of layers that compute the weighted sum of the inputs, which is then passed through a non-linearity to create the output. Technically, you can use whatever shape or form to express a problem(as long as the input and output dimensions match your problem description and it has at least one hidden layer). However, that's similar to asking the actors on stage to perform while all wearing similarly coloured costumes with a bare-minimum prop set and a blank background. With a lot of focus and rehearsal, it might just work out, but there must be a better way of doing things. 

Let's revisit the question: 

>"How do you design a model such that, despite the limited context of weighted sums, manages to capture the key properties of the given data?"*

That should be part of our active considerations when implementing a neural network. A successful example is the introduction of Convolutional Neural Networks(CNN) for image data. By utilising the weighted sums in 2D convolution kernels, it's been shown to better capture the *spatial dependencies* that we expect the pixels to have, compared to flattening the image and then using a dense multi-layer 1D perceptron model. By giving the actors specific cues on where to stand as they interact, the stage is suddenly flowing more naturally, and the true shape of the play starts to emerge more vividly.

In sequential data, the critical property to consider is the ability to utilise past data in the current prediction, i.e. to model the *temporal dependency*. Recurrent Neural Networks(RNN), Long Short Term Memory(LSTM), and Gated Recurrent Units(GRU) are cases where their formulation helps weigh between the past context to the current input. They eventually boil down to structures that produce learned weights representing the suitable ratio of what to forget and what to remember between the past and present at the current timestep.


{% include figure image_path="/assets/positional_encoding/nn.png" alt="neural network diagrams" caption="*This diagram shows a very small part of a neural network model, and it serves only to demonstrate attempts that modified the weighted sum concept to accomodate specific properties*"%}


Like these examples, mathematical modelling, including neural networks, is all about capturing the interaction happening on stage with multiplication, addition, and composition/convolution. To do so effectively, **we build instructions that can portray the key properties we'd like to extract.**



## Attention

The attention mechanism from computer vision
problems, and natural language understanding, to sequence modelling and
protein modelling, holds the same principles as described in the
introduction. A well-trained attention layer aims to find suitable
weights that signify the relative importance of elements in a
**sequence** given the query of a specific timestep of the sequence. The
weighted sum of the values representing each sequence step encapsulates the information embedded in the query
timestep (or at least we hope it does). This is different from previous
methods of modelling sequential data with neural networks such as RNNs
and LSTMs, as they parse each sequence step in order, unlike the
parallel nature of attention mechanisms.


A scoring function generates the weights holding the relative importance
of each value, such as a scaled dot product between the query and a key.
As you can see, the query, key, and value are vectors generated for each
sequence element used for attention calculations. They are created from
separate learned weights that transform the input accordingly. With that
out of the way, we can pose this question: "If an element has a key
vector that does not depend on its position within the sequence, then
surely the eventual value weights will have no difference from that
element appearing at different timesteps?\" The scoring function taking
in the query and the key will see no difference between taking the same
key at the beginning or the end of the sequence. This is not a problem
for static sequences with no meaning in their order/position, but it is
a problem for sequences that carry meaning in their position, such as
text, audio, and video. (e.g. The same noun can either be a subject or
an object depending on its location within a sentence). RNNs and LSTMs
inherently take the position of each element in their design as data is
processed step-by-step, yet the parallel and scalable nature of
attention are what we want to use. Handling this problem is where
positional encodings come in.



