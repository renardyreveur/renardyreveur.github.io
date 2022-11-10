---
layout: post
title:  "Sinusoidal and Rotary Positional Encodings"
date:   2022-11-07 15:15:52 +0900
# categories: attention transformers deep-learning positional-encodings
---

# Introduction

#### 

Let's face it; the world is too complicated. Things change over time,
multiple variables are associated with any given event or phenomenon,
and the layers of abstraction built up are dizzying. In our attempts to
make sense of this complex, dynamic world around us, we seek to find
*relationships between these variables*, uncovering the underlying
patterns embedded in those abstractions.

#### 

Deep Learning is a method that attempts to uncover these patterns by
learning from data. We represent a given problem with the data that it
generates. By fitting the data to a neural network structure, we find
how the dimensions within the data are interconnected. The way that deep
learning captures that relationship is quite simple; all you have to do
is look at how a neural network is made up. The perceptron, the building
blocks of neural networks, are linear combinations of their inputs with
some non-linearity injected in. Thus, at the most basic level, we are
modelling the inter-dependencies between given variables with their
weighted sums. For example, convolutional neural networks(CNN) slide
multiple kernels of learnable weights over the 2D pixel array producing
a weighted sum between the kernel and windowed pixel area to find local
spatial patterns. Recurrent neural networks(RNN) and Long Short Term
Memory(LSTM) networks constantly weigh between the current step input
and the previous state to find temporal patterns. And on and on. We're
going to focus on this notion of weighted sums in this article as we
explore the idea of positional encoding.

# Attention

#### 

The attention mechanism, now being used everywhere, from computer vision
problems, and natural language understanding, to sequence modelling and
protein modelling, holds the same principles as described in the
introduction. A well-trained attention layer aims to find suitable
weights that signify the relative importance of elements in a
**sequence** given the query of a specific timestep of the sequence. The
weighted sum (using the weights from above) of the values representing
each sequence step encapsulates the information embedded in the query
timestep (or at least we hope it does). This is different from previous
methods of modelling sequential data with neural networks such as RNNs
and LSTMs, as they parse each sequence step in order, unlike the
parallel nature of attention mechanisms.

#### 

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

# Positional Encoding

#### 

Let us define the problem of including positional information in
parallel sequence modelling. What do we want? We want the scoring
function to acknowledge the difference between different positions
within a sequence. On top of this, we can extend the condition to have
the scoring function understand the relative difference in position
between the query and the key. Position is an absolute and relative
concept, which is exactly what we want the scoring function to
understand.

#### 

The simplest way of encoding position is to label each position with its
index in the sequence, adding/concatenating this information to the
query and keys. With this, the key for the same token(word/subword/etc)
at different positions will be different. This is a good start, but it's
easy to notice a significant limitation of this approach. As the
sequence gets longer, the label that will be either added or
concatenated to the key/query vectors will become larger and larger.
Large indices are likely to play poorly when deep neural networks are
susceptible to unstable training depending on numeric scale.

#### 

Can't we just normalize it? somebody shouts from the back. Sure, that is
a way forward in solving the scale problem, but how are you going to
normalize it? If it depends on the sequence length, that means position
2 of length 5 sequence and position 2 of a length 100 sequence will have
different position encodings! We can't have that! The same position,
regardless of length of the original sequence has to have the same
encoding for the model.

#### 

It becomes evident that using a single number to encode the unique
position of each token in a sequence might not be enough. Consider using
the binary representation of each index as a position encoding
**vector**; the numbers are either 0 or 1, so that helps with the scale
problem, and it isn't dependent on the sequence length! This is good,
we're opening up to using vectors(a set of numbers) to represent
position, and it gives a unique representation at each step. But a small
caveat appears out of nowhere, and it is the fact that we are using a
discrete function to represent the binary numbers. To understand the
problem more intuitively, consider the example of a 2-dimensional
positional encoding vector capable of portraying the positions of a
sequence with max length 4. The positional encoding vectors for each
position are as follows: $[0, 0], [0, 1], [1, 0], [1, 1]$. Estimating
intermediate position values is hard as the function mapping on 2d space
is not an injective nor a continuous function. It'd be better to have a
proper function mapping the values of the vector given its scalar order,
namely a discretization of continuous function.

#### 

The function that the original authors of the 'Attention is all you
need' paper decided to use is the sine function. The binary
representation of the index has alternating zeros and ones but at
different periods for each dimension, making them unique. This
periodicity, I guess, naturally led the authors to the sine function
with the continuously increasing period over the dimensions. i.e.
creating a vector with the form:

$$PE_{i, j} = sin \left( \frac{1}{\omega^{j/d_{model}}} \ i \right)$$

where $i$ represents the index of the sequence, and $j$ the dimension of
the positional encoding vector. The length of the positional encoding
vector is set to be the same as the dimension that the rest of the model
uses for ease in adding the positional encoding to the query and key
vectors. Thus, $j/d_{model}$ varies between 0 and 1. The first dimension
of the positional encoding vector will have a minimum period of $2\pi$
radians, and the last dimension will have a maximum period of
$\omega * 2\pi$ radians. The authors set $\omega$ as $10000$, probably a
choice made after experimentation.

#### 

Wow, we're finally here, the holy grail of encoding absolute positions
with a vector. We've found something that uniquely encodes the position
of each element in a sequence, satisfying most of the conditions we've
come across so far. We could end it here, but remember that additional
property we said we would like the positional encoding to have?: To be
able to encode the relative difference in position as well. Remember how
the attention mechanism is basically creating a linear combination
between the tokens as a form of sequence modelling? Suppose the position
encoding vectors (which are injected into the keys and queries) are
linearly dependent in the sense that linear combinations of position
vectors can translate the position vector to different positions. In
that case, this will also match the linear combination of tokens in the
sequence, attending to the relative position as we wanted.

#### 

The extension comes by selecting rotation as the linear transformation
that translates between positions. If we use pairs of sine and cosines
instead of just sines, we can create a rotation matrix that rotates the
pairs in the position vector by a certain angle. The relative position
between the query and the key determines the angle. Woop woop! We're
here!

#### 

The final positional encoding vector is as follows:

$$\begin{aligned}
        PE_{i, 2j} &= sin \left( \frac{1}{\omega^{2j/d_{model}}} \ i \right) \\
        \\
        PE_{i, 2j+1} &= cos \left( \frac{1}{\omega^{2j/d_{model}}} \ i \right)
    
\end{aligned}$$

Pairs of sine and cosine with the same period are used, and the period
monotonically increases as you go down the pairs in dimension.

::: center
![image](/positional-encoding/assets/positional_encoding/sin_pos_enc.png)
:::

# Rotary Positional Encoding - RoPE

#### 

Sinusoidal absolute positional encoding is one of many ways to encode a
sequence's position. Numerous research has been done since the original
Transformer paper, coming up with methods such as T5 or Transformer XL's
relative positional bias or even learnable embeddings, unlike the fixed
versions we've discussed.

#### 

Relative encodings are based on creating a pairing of positions in the
sequence and then encoding the relative difference between them. The
pairing is typically expressed with a $N$ x $N$ matrix between
positions, which might not work with efficient transformer
implementations. The point is there needs to be a more principled
approach to coming up with these encodings, and RoPE is one of them.

#### 

The good people at EleutherAI developed a principled approach to the
relative positional encoding that is generally applicable and works for
'efficient' variants of attention. Given the dot product as the scoring
function between the query and keys, enabling knowledge conveyance
between tokens at different positions, they create a constraint on the
inner product of the query and key that incorporates relative positional
information.

#### 

We want to find a **positional encoding function** $f(\mathbf{x}, l)$
for token $\mathbf{x}$ and its position $l$ such that for two items
$\mathbf{x_q}$ and $\mathbf{x_k}$ at positions $m$ and $n$, it satisfies
the following:

$$\left < f(\mathbf{x_q}, m), f(\mathbf{x_k}, n) \right > = g(\mathbf{x_q}, \mathbf{x_k}, n-m)$$

#### 

That is, the inner product between two position-encoded vectors should
only be dependent on the original vectors and their relative positional
differences. The inner product of the position encoded vectors is the
score function, and it makes sense to only depend on the original token
values and their relative positional difference. Shifting the query and
key position by the same amount will change their absolute position but
not relative position. We need to find a function that allows for the
dot product in the attention mechanism to have this property: preserving
the relative positional information while discarding the absolute
position. Recall that the geometric interpretation of the dot product is
given as
$$\mathbf{x} \cdot \mathbf{y} = ||\mathbf{x}|| ||\mathbf{y}|| \cos \theta$$
where $\theta$ is the angle between the vectors. **If we can represent
the positional information as rotations applied to the vectors**, it
satisfies the preservation of relative differences as if the same angle
rotates both vectors, their dot product won't change.

#### 

To find the function that maps the positional information to rotations,
we first change our view of the token vector to be a vector of complex
numbers made with pairs of the original vector. That is, we view

$$\mathbf{x_q} = (x_{q1}, x_{q2}, \dots, x_{qd}) \in \mathbb{R}^d$$

As

$$\mathbf{x_q} = (x_{q1} + ix_{q2}, x_{q3} + ix_{q4}, \dots, x_{qd-1} + ix_{qd}) \in \mathbb{C}^{d/2}$$

#### 

In this representation, the positional encoding functions become:

$$\begin{aligned}
        f(\mathbf{x_q}, m) &= R_{f(\mathbf{x_q}, m)}e^{i\Theta_{f(\mathbf{x_q}, m)}} \\
        f(\mathbf{x_k}, m) &= R_{f(\mathbf{x_k}, n)}e^{i\Theta_{f(\mathbf{x_k}, n)}} \\
        g(\mathbf{x_q},\mathbf{x_k}, n-m) &= R_{g(\mathbf{x_q},\mathbf{x_k}, n-m)}e^{i\Theta_{g(\mathbf{x_q},\mathbf{x_k}, n-m)}} \\
    
\end{aligned}$$

#### 

The inner product with this representation yields the following
equations:

$$\begin{aligned}
        R_{f(\mathbf{x_q}, m)}R_{f(\mathbf{x_k}, n)} &= R_{g(\mathbf{x_q},\mathbf{x_k}, n-m)} \\
        \Theta_{f(\mathbf{x_q}, m)} - \Theta_{f(\mathbf{x_k}, n)} &= \Theta_{g(\mathbf{x_q},\mathbf{x_k}, n-m)}
    
\end{aligned}$$

#### 

This is from the fact that given two complex numbers $z_1$ and $z_2$,
the inner product is given by
$z_1 \cdot z_2 = |z_1||z_2| \cos(\theta_1 - \theta_2)$ where $\theta_1$
and $\theta_2$ are the angles of the complex numbers.

#### 

By setting some intial condition $f(\mathbf{x}, 0) = \mathbf{x}$, we
notice that

$$R_{f(\mathbf{x_q}, 0)}R_{f(\mathbf{x_k}, 0)} = R_{g(\mathbf{x_q},\mathbf{x_k}, 0)} = \mathbf{x_q}\mathbf{x_k}$$

#### 

If $R_{g(\mathbf{x_q},\mathbf{x_k}, 0)} = \mathbf{x_q}\mathbf{x_k}$ Then
for all $m = n$ we have that
$R_{f(\mathbf{x_q}, m)}R_{f(\mathbf{x_k}, m)} = \mathbf{x_q}\mathbf{x_k}$
as well as $m-m = 0$

#### 

This means that $R$ is indpendent of the position $m$, and ths we can
simply say $$R_{f(\mathbf{x}, y)} = \mathbf{x}$$ for all position $y$.

#### 

Similarly, by setting the initial condition
$\Theta(\mathbf{x}) = \Theta(\mathbf{x}, 0)$, we have:

$$\Theta_{f(\mathbf{x_q}, 0)} - \Theta_{f(\mathbf{x_k}, 0)} = \Theta(\mathbf{x_q}) - \Theta(\mathbf{x_k})$$

#### 

Utilizing the same logic as the $R$ case, we can see that
$\Theta_{f(\mathbf{x_q}, m)} - \Theta(\mathbf{x_q}) = \Theta_{f(\mathbf{x_k}, m)} - \Theta(\mathbf{x_k})$
for all $\mathbf{x_q}$, $\mathbf{x_k}$. It doesn't depend on the query
or key. Denoting $\Theta_{f(x, m)} - \Theta(x)$ as $\varphi(m)$, we have
the following useful representation of $\Theta_{f(x, m)}$:

$$\Theta_{f(x, m)} = \Theta(\mathbf{x}) + \varphi(m)$$

#### 

Plugging this in to the original equation with m = n+1, we have:
$$\begin{aligned}
        \Theta_{f(\mathbf{x_q}, m)} - \Theta_{f(\mathbf{x_k}, m+1)} &= \Theta_{g(\mathbf{x_q},\mathbf{x_k}, 1)} \\
        &= \Theta(\mathbf{x_q}) + \varphi(m) - \Theta(\mathbf{x_k}) - \varphi(m+1)\\
        \\
        \varphi(m+1) - \varphi(m) &= \Theta_{g(\mathbf{x_q},\mathbf{x_k}, 1)} + \Theta(\mathbf{x_q}) - \Theta(\mathbf{x_k}) \\
    
\end{aligned}$$

#### 

Interestingly, the RHS of the equation does not depend on $m$ and thus
$\varphi$ must also not depend on $m$, rendering it down to an
arithmetic progression. If we set initial values $\varphi(0) = 0$ and
$\varphi(1) = \theta$, where $\theta$ is a step rotation amount to be
decided, we have $\varphi(m) = m\theta$

#### 

Plugging this back into the original equation, we have:
$$\Theta_{f(x, m)} = \Theta(\mathbf{x}) + m\theta$$

#### 

We now have all the ingredients to write the positional encoding
functions in the complex form:

$$\begin{aligned}
        f(\mathbf{x_q}, m) &= R_{\mathbf{x_q}}e^{i(\Theta(\mathbf{x_q})+ m\theta)} \\
        &= ||\mathbf{x_q}||e^{i(\theta_q + m\theta)} \\
        &= ||\mathbf{x_q}||e^{i\theta_q}e^{im\theta} \\
        &= \mathbf{x_q}e^{im\theta} \\
        \\
        f(\mathbf{x_k}, m) &= R_{\mathbf{x_k}}e^{i(\Theta(\mathbf{x_k})+ m\theta)} \\
        &= ||\mathbf{x_k}||e^{i(\theta_k + m\theta)} \\
        &= ||\mathbf{x_k}||e^{i\theta_k}e^{im\theta} \\
        &= \mathbf{x_k}e^{im\theta} \\
    
\end{aligned}$$

#### 

What a derivation that was! We arrived at our destination; we have a
positional encoding function that takes in the token vectors and gives
us a position-encoded version that, when dot-producted with another
position-encoded vector, will preserve the relative positional
difference information.

#### 

To use it, it'd be best to change the view to the real version. As an
intuitive example, let us reduce the problem down to the 2d case where

$$\mathbf{x_q} = (x_{q1}, x_{q2}) \in \mathbb{R}^2$$
$$\mathbf{x_q} = (x_{q1} + ix_{q2}) \in \mathbb{C}$$

#### 

Here, applying the positional encoding function will look like the
following.

#### 

Recall the positional encoding function:
$$f(\mathbf{x}, m) = \mathbf{x}e^{im\theta}$$

#### 

Expand the exponential using Euler's identity:
$$f(\mathbf{x}, m) = \mathbf{x}e^{im\theta} = \mathbf{x}(cos(m\theta) + isin(m\theta))$$

#### 

Applying this function to the 2D case will look like:

$$\begin{aligned}
        f(\mathbf{x_q}, m) &= (x_{q1} + ix_{q2})e^{im\theta} \\
        &= (x_{q1}+ ix_{q2})(cos(m\theta) + isin(m\theta)) \\
        &= x_{q1}cos(m\theta) + ix_{q2}cos(m\theta) + ix_{q1}sin(m\theta) - x_{q2}sin(m\theta) \\
        &= x_{q1}cos(m\theta) - x_{q2}sin(m\theta) + i(x_{q1}sin(m\theta) + x_{q2}cos(m\theta)) \\
    
\end{aligned}$$

#### 

The real and imaginary parts of the complex number represent a pair of
dimensions in the original 'real' case. We can see that the positional
encoding function modified the initial token vector as:

$$\begin{aligned}
        \begin{bmatrix}
            x_{q1} \\
            x_{q2}
        \end{bmatrix}
        \rightarrow
        \begin{bmatrix}
            x_{q1}cos(m\theta) - x_{q2}sin(m\theta) \\
            x_{q1}sin(m\theta) + x_{q2}cos(m\theta)
        \end{bmatrix}
    
\end{aligned}$$

This can be written as a matrix multiplication as well:
$$\begin{bmatrix}
        x_{q1} \\
        x_{q2}
    \end{bmatrix}
    \rightarrow
    \begin{bmatrix}
        cos(m\theta) & -sin(m\theta) \\
        sin(m\theta) & cos(m\theta)
    \end{bmatrix}
    \begin{bmatrix}
        x_{q1} \\
        x_{q2}
    \end{bmatrix}$$

#### 

This is precisely the rotation matrix we were talking about earlier! We
managed to represent positional information as rotations applied to
vectors!

#### 

We can generalize this to the $d$-dimensional case by applying the above
to each pair of dimensions of the token vector.

::: center
![It's hard to show the position encoded vector as RoPE is a function
that applies rotations to the token vector. So I just added a picture of
a random query vector that has gone through RoPE](/positional-encoding/assets/positional_encoding/rotary.png)
:::

#### 

This may seem similar to the absolute sinusoidal positional encoding
function, but it is not. The absolute sinusoidal positional encoding
function is a function only of the token's position in the sequence. In
contrast, the relative positional encoding function is a function of the
token itself and the position of the token in the sequence. On top of
this, while the fixed absolute positional encoding is added to the token
vector, RoPE is multiplied based on the first-principle derivation we
went through above.

#### 

The Rotary Positional Encoding idea has seen a lot of success in many
Pre-trained Language Models and is generating quite a buzz in the
Chinese NLP community. There are more and more implementations of PLTs
with RoPE, and I believe we'll see more of them in the future.

#### 

What I like about RoPE is that it formulated the problem of positional
encoding in a principled manner, allowing us to see why this might work
and how this has come to be. Approaches with strong foundations like
this is not usually seen in the deep learning community, where telltales
of experiment results are often used to justify the design of a model.

# References {#references .unnumbered}

1.  *Attention is all you need* <https://arxiv.org/abs/1706.03762>

2.  *Roformer - paper that introduced RoPE*
    <https://arxiv.org/abs/2104.09864>

3.  *EleutherAI blogpost on RoPE*
    <https://blog.eleuther.ai/rotary-embeddings/>

4.  *Jonathan Kernes' blogpost on Positional Encoding*
    <https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3>

5.  *Kazemnejad's blogpost on Positional Encoding*
    <https://kazemnejad.com/blog/transformer_architecture_positional_encoding/>
