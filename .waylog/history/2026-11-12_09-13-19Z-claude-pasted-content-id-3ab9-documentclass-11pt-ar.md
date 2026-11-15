# <pasted_content id="3ab9">
\documentclass[11pt]{ar
_Exported on 11/12/2026 at 03:13:50 GMT-6 from Claude via WayLog_


**User**

<pasted_content id="3ab9">
\documentclass[11pt]{article}

\usepackage[margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc}
\usepackage{caption}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}

\captionsetup{font=small,labelfont=bf}

\tikzset{
  box/.style={draw,rounded corners=2pt,align=center,font=\footnotesize,minimum height=7mm,inner sep=3pt,fill=white},
  inp/.style={box,fill=blue!6},
  op/.style={box,fill=gray!8},
  out/.style={box,fill=green!8},
  arr/.style={-{Stealth[length=2mm]},thick},
}

% Notation
\newcommand{\R}{\mathbb{R}}
\newcommand{\regions}{\mathcal{R}}
\newcommand{\vc}{\mathbf{c}}
\newcommand{\vchat}{\hat{\mathbf{c}}}
\newcommand{\ve}{\mathbf{e}}
\newcommand{\vq}{\mathbf{q}}
\newcommand{\vs}{\mathbf{s}}
\newcommand{\vh}{\mathbf{h}}
\newcommand{\vy}{\hat{\mathbf{y}}}
\newcommand{\vb}{\mathbf{b}}
\newcommand{\vu}{\mathbf{u}}
\newcommand{\mZ}{\mathbf{Z}}
\newcommand{\mX}{\mathbf{X}}
\newcommand{\mS}{\mathbf{S}}
\newcommand{\mW}{\mathbf{W}}
\newcommand{\GELU}{\operatorname{GELU}}
\newcommand{\ReLU}{\operatorname{ReLU}}
\newcommand{\LN}{\operatorname{LN}}
\newcommand{\Drop}{\operatorname{Drop}}
\newcommand{\softmax}{\operatorname{softmax}}

\title{ContextPose}
\author{}
\date{}

\begin{document}
\maketitle

\section{Methodology}
\label{sec:method}

ContextPose predicts the valence and arousal that annotators assign to a clip of a child around an environmental event. The model is built on one idea: \emph{the event context decides which moments of the child's behaviour matter}. Rather than appending the context to a pooled pose representation, ContextPose uses the context to form the query of an attention operation over time. The context therefore selects which time steps of each window are summarised. The architecture (Figure~\ref{fig:arch}) has five components:
\begin{enumerate}
  \item a temporal convolutional pose encoder shared across windows (\S\ref{sec:pose-encoder});
  \item a context encoder that maps the event description to the model width (\S\ref{sec:context});
  \item context-queried cross-attention that pools each window into one vector (\S\ref{sec:cross-attention});
  \item an inter-region Transformer and a prediction head that relate the pre-event, event and post-event windows (\S\ref{sec:region-transformer});
  \item an optional gated face stream that adds a residual correction (\S\ref{sec:face}).
\end{enumerate}
The whole network predicts a correction to a fixed, context-dependent offset (\S\ref{sec:residual}).

%-----------------------------------------------------------------------------
\begin{figure}[t]
\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[every node/.style={text width=1.75cm}]
  % Pose columns
  \foreach \name/\x/\lbl in {pre/0/pre-event, ev/3/event, post/6/post-event}{
    \node[inp] (P\name) at (\x,0) {Pose $\mathbf{P}_{\text{\lbl}}$\\[-1pt]{\scriptsize $T{\times}J{\times}F$}};
    \node[op]  (T\name) at (\x,-1.4) {Temporal conv.\\[-1pt]{\scriptsize $\to \mZ_r\in\R^{T\times W}$}};
    \node[op,minimum height=11mm] (A\name) at (\x,-2.9) {Cross-attn.\\[-1pt]{\scriptsize $\vq_r=\vchat+\ve_r$}};
    \draw[arr] (P\name) -- (T\name);
    \draw[arr] (T\name) -- node[right,font=\scriptsize,text width=]{$\mZ_r$} (A\name);
  }

  % Context path
  \node[inp,text width=2.3cm] (C) at (-2.9,1.35) {Context $\vc$\\[-1pt]{\scriptsize indicators / CLIP / VLM}};
  \node[op] (Wc) at (-0.15,1.35) {Linear\\[-1pt]{\scriptsize $\vchat=\mW_c\vc+\vb_c$}};
  \draw[arr] (C) -- (Wc);
  \draw[thick] (Wc.east) -- (7.25,1.35);
  \foreach \name/\x in {pre/1.25, ev/4.25, post/7.25}{
    \draw[arr] (\x,1.35) |- (A\name.east);
  }

  % Region transformer and head
  \node[op,text width=7.6cm] (RT) at (3,-4.6) {Inter-region Transformer over $[\vs_{\text{pre}};\vs_{\text{event}};\vs_{\text{post}}]$, then mean over regions $\to \vh$};
  \foreach \name in {pre,ev,post}{ \draw[arr] (A\name) -- node[right,font=\scriptsize,text width=]{$\vs_r$} (A\name |- RT.north); }
  \node[op,text width=3.2cm] (H) at (3,-5.85) {MLP head $W\!\to\!W/2\!\to\!2$};
  \draw[arr] (RT) -- (H);
  \node[draw,circle,inner sep=1pt,text width=,font=\small] (plus) at (3,-7.0) {$+$};
  \draw[arr] (H) -- node[right,font=\scriptsize,text width=]{$\vy_{\text{pose}}$} (plus);
  \node[inp,text width=2.6cm] (bg) at (-0.2,-7.0) {Offset $\vb_{g(\vc)}$\\[-1pt]{\scriptsize context lookup}};
  \draw[arr] (bg) -- (plus);
  \node[out,text width=2.6cm] (y) at (3,-8.1) {$\vy=[\hat y_{\text{val}},\hat y_{\text{aro}}]$};
  \draw[arr] (plus) -- (y);

  % Face stream
  \node[inp,text width=2.4cm] (F) at (9.7,0) {Face features $\mathbf{x}^{f}_{r,t}$, gate $m_{r,t}$};
  \node[op,text width=2.4cm] (FE) at (9.7,-1.4) {LN $\to$ Linear $\to$ GELU\\[-1pt]{\scriptsize $\vu_{r,t}\in\R^{32}$}};
  \node[op,text width=2.4cm,minimum height=11mm] (FP) at (9.7,-2.9) {Gated mean per region\\[-1pt]{\scriptsize $\bar{\vu}_r$}};
  \node[op,text width=2.4cm] (FH) at (9.7,-4.6) {Face head $g$\\[-1pt]{\scriptsize zero-initialised}};
  \draw[arr] (F) -- (FE);
  \draw[arr] (FE) -- (FP);
  \draw[arr] (FP) -- (FH);
  \draw[arr] (FH.south) |- node[pos=0.75,above,font=\scriptsize,text width=]{$\bm{\Delta}_{\text{face}}$} (plus.east);
\end{tikzpicture}}
\caption{ContextPose. Each window's pose sequence is encoded by a temporal convolution; the convolution and the cross-attention share their weights across the three windows. A query formed from the projected context and a learned region embedding attends over the window's time steps. The three window summaries are related by a Transformer layer, averaged, and decoded to valence and arousal. An optional gated face stream adds a zero-initialised correction. The output is a residual on a context-lookup offset. In the FiLM variant (\S\ref{sec:film}) the context modulates $\mZ_r$ instead, and the query is $\ve_r$ alone.}
\label{fig:arch}
\end{figure}
%-----------------------------------------------------------------------------

\subsection{Problem formulation and inputs}
\label{sec:inputs}

Each clip is split into three windows around the event, indexed by $r\in\regions=\{\text{pre},\text{event},\text{post}\}$: the 5\,s before the event, the event itself, and the 5\,s after it. For each clip the model receives:
\begin{itemize}
  \item a pose sequence $\mathbf{P}_r\in\R^{T\times J\times F}$ per window;
  \item a context vector $\vc\in\R^{d_c}$ that describes the event;
  \item optionally, per-step face features $\mathbf{X}^f_r\in\R^{T\times D}$ with a binary quality mask $\mathbf{m}_r\in\{0,1\}^{T}$.
\end{itemize}
It outputs two unconstrained scores $\vy\in\R^2$, one for valence and one for arousal.

\paragraph{Pose representation.} We use $J=11$ upper-body COCO keypoints: the nose, eyes, ears, shoulders, elbows and wrists. Let $\mathbf{p}_{n,j}\in\R^2$ be the image position of joint $j$ in frame $n$. Positions are centred on the shoulder midpoint $\mathbf{o}_n$ and scaled by the clip's median shoulder width $w$:
\begin{equation}
  \tilde{\mathbf{p}}_{n,j} = \frac{\mathbf{p}_{n,j}-\mathbf{o}_n}{w}.
\end{equation}
Each window, whatever its real duration, is resampled linearly to $T=32$ steps. Each step carries $F=4$ features per joint: the normalised coordinates, the detector confidence, and the joint speed in shoulder widths per second. The speed is measured on the original $f_s=25$\,fps frames, before resampling:
\begin{equation}
  v_{n,j} = \min\!\left(f_s\,\lVert\tilde{\mathbf{p}}_{n,j}-\tilde{\mathbf{p}}_{n-1,j}\rVert_2,\; v_{\max}\right), \qquad v_{\max}=20.
\end{equation}
It is then averaged over the frames that fall in each resampling bin. Resampling changes the apparent frame-to-frame displacement, so it no longer reflects real speed. The speed channel keeps that information in real-time units. The cap $v_{\max}$ suppresses tracking jumps.

\subsection{Temporal pose encoder}
\label{sec:pose-encoder}

The joints of each step are flattened into one vector, $\mX_r\in\R^{T\times JF}$. A two-layer temporal convolution with kernel size $3$ and zero padding maps $\mX_r$ to step tokens of width $W$:
\begin{equation}
  \mZ_r = \GELU\!\big(\mathrm{Conv}_2\big(\GELU(\mathrm{Conv}_1(\mX_r))\big)\big)\in\R^{T\times W},
  \qquad
  \big[\mathrm{Conv}(\mX)\big]_t = \vb + \sum_{k=-1}^{1}\mW_k\,\mathbf{x}_{t+k},
\end{equation}
with $\mathbf{x}_0=\mathbf{x}_{T+1}=\mathbf{0}$, $\mW_k\in\R^{W\times JF}$ in the first layer and $\mW_k\in\R^{W\times W}$ in the second. Each token $\mathbf{z}_{r,t}$ therefore summarises a local neighbourhood of five steps. The encoder is shared across the three windows. We add no positional encoding: temporal order enters the tokens only through the local receptive field of the convolution.

\subsection{Context encoder}
\label{sec:context}

The context vector $\vc$ takes one of three forms:
\begin{itemize}
  \item the four binary event indicators (social interaction, auditory event, activity or object change, visual or media event), $d_c=4$;
  \item the frozen, $\ell_2$-normalised CLIP ViT-L/14 embedding of a sentence that describes the event, $d_c=768$;
  \item the $\ell_2$-normalised sentence embedding of a vision-language model's description of the clip, $d_c=384$.
\end{itemize}
A linear layer projects every form to the model width:
\begin{equation}
  \vchat = \mW_c\,\vc + \vb_c\in\R^{W}.
\end{equation}
Each window $r$ also has a learned region embedding $\ve_r\in\R^W$, initialised from $\mathcal{N}(0,0.02^2)$. The region embedding tells the shared attention which window it is reading.

\subsection{Context-queried temporal attention}
\label{sec:cross-attention}

The core of ContextPose is a cross-attention whose query depends on the context and on the window, and whose keys and values are the pose tokens of that window. The query for window $r$ is
\begin{equation}
  \vq_r = \vchat + \ve_r\in\R^{1\times W}.
\end{equation}
With $H=4$ heads of width $d_h=W/H$, head $i$ computes
\begin{align}
  \mathbf{Q}_{r,i} &= \vq_r\mW^{Q}_i, \qquad
  \mathbf{K}_{r,i} = \mZ_r\mW^{K}_i, \qquad
  \mathbf{V}_{r,i} = \mZ_r\mW^{V}_i, \\
  \bm{\alpha}_{r,i} &= \softmax\!\left(\frac{\mathbf{Q}_{r,i}\mathbf{K}_{r,i}^{\top}}{\sqrt{d_h}}\right)\in\R^{1\times T}, \label{eq:attn}
\end{align}
and the heads are concatenated and projected:
\begin{equation}
  \vs_r = \big[\bm{\alpha}_{r,1}\mathbf{V}_{r,1}\;\Vert\;\cdots\;\Vert\;\bm{\alpha}_{r,H}\mathbf{V}_{r,H}\big]\,\mW^{O}\in\R^{W}.
\end{equation}
Here $\mW^{Q}_i,\mW^{K}_i,\mW^{V}_i\in\R^{W\times d_h}$ and $\mW^{O}\in\R^{W\times W}$. Biases are omitted for brevity. The attention weights are shared across the three windows.

Equation~\eqref{eq:attn} makes the role of the context explicit. Each summary $\vs_r$ is a convex combination of projected step tokens, and the weights $\bm{\alpha}_{r,i}$ depend on $\vc$. The context does not add information to the pose features. It decides \emph{which time steps} of each window contribute to the summary. For example, a social-interaction event may draw attention to a different part of the post-event window than an auditory event does. Because the query has a single token, each window is reduced to one vector, whatever the duration of the event.

\subsubsection{FiLM conditioning variant}
\label{sec:film}

As an alternative to querying, the binary indicators can modulate the pose tokens through feature-wise linear modulation (FiLM). A two-layer network $f_c$ predicts a scale and a shift from the context:
\begin{equation}
  [\bm{\gamma};\bm{\beta}] = f_c(\vc) = \mW_{2}\,\GELU(\mW_{1}\vc+\vb_{1})+\vb_{2}\in\R^{2W},
\end{equation}
with a hidden width of 64. These modulate every token of every window:
\begin{equation}
  \mZ'_r = (\mathbf{1}+\bm{\gamma})\odot\mZ_r + \bm{\beta},
\end{equation}
broadcast over the $T$ steps. The query is then the region embedding alone, $\vq_r=\ve_r$, so the context reaches the output only through the modulation. The last layer of $f_c$ is zero-initialised ($\mW_2=\mathbf{0}$, $\vb_2=\mathbf{0}$), so training starts from the identity, $\mZ'_r=\mZ_r$.

\subsection{Inter-region Transformer and prediction head}
\label{sec:region-transformer}

The three window summaries form a short sequence, $\mS=[\vs_{\text{pre}};\vs_{\text{event}};\vs_{\text{post}}]\in\R^{3\times W}$. One post-norm Transformer encoder layer with $H=4$ heads relates them:
\begin{align}
  \mS' &= \LN\!\big(\mS + \Drop(\mathrm{MHSA}(\mS))\big), \\
  \mS'' &= \LN\!\big(\mS' + \Drop(\mathrm{FFN}(\mS'))\big), \qquad
  \mathrm{FFN}(\mathbf{x}) = \mW_4\,\Drop\!\big(\ReLU(\mW_3\mathbf{x}+\vb_3)\big)+\vb_4,
\end{align}
where $\mathrm{MHSA}$ is multi-head self-attention over the three regions and the feed-forward width is $2W$. This layer lets the model compare the child's behaviour before, during and after the event, for example a change in movement after the event. The output is averaged over the regions:
\begin{equation}
  \vh = \frac{1}{3}\sum_{r\in\regions}\mS''_r\in\R^{W}.
\end{equation}
An optional \emph{context skip} also adds the projected context to the fused vector, $\vh\leftarrow\vh+\vchat$. This gives the context a direct path to the head, in addition to its effect through the attention weights.

A two-layer MLP with dropout on its input decodes the fused vector:
\begin{equation}
  \vy_{\text{pose}} = \mW_6\,\GELU\!\big(\mW_5\,\Drop_{p_h}(\vh)+\vb_5\big)+\vb_6\in\R^{2},
\end{equation}
with hidden width $W/2$.

\subsection{Gated face stream}
\label{sec:face}

Facial expression is a direct cue to valence, but face crops of children in unconstrained video are often unusable: the head is turned or lowered, hidden by hair, dark or blurred. The face stream is therefore a separate branch that adds a correction to the pose prediction and relies only on gated steps.

At each step, $\mathbf{x}^f_{r,t}\in\R^{D}$ is the $D=768$-dimensional [CLS] embedding of a frozen facial-expression ViT applied to the child's face crop. The mask $m_{r,t}\in\{0,1\}$ is $1$ only if the step passes a quality gate:
\begin{itemize}
  \item the face is frontal (nose and both eyes tracked);
  \item a face detector confirms a face at the tracked location;
  \item the crop's brightness and sharpness exceed fixed thresholds.
\end{itemize}

Each step is encoded with a normalised linear layer:
\begin{equation}
  \vu_{r,t} = \GELU\!\big(\mW_f\,\LN(\mathbf{x}^f_{r,t})+\vb_f\big)\in\R^{32}.
\end{equation}
The encodings are averaged over the gated steps of each window:
\begin{equation}
  \bar{\vu}_r = \frac{\sum_{t=1}^{T} m_{r,t}\,\vu_{r,t}}{\max\!\big(1,\sum_{t=1}^{T} m_{r,t}\big)}.
\end{equation}
A two-layer head $g$ ($96\to W/2\to k$) maps the concatenated window encodings to a correction:
\begin{equation}
  \bm{\delta} = g\big(\Drop_{p_h}([\bar{\vu}_{\text{pre}};\bar{\vu}_{\text{event}};\bar{\vu}_{\text{post}}])\big)\in\R^{k},
  \qquad
  \bm{\Delta}_{\text{face}} = \mathbb{I}\Big[\textstyle\sum_{r,t} m_{r,t}>0\Big]\cdot
  \begin{cases}
    [\delta_1,\,0]^{\top}, & k=1 \text{ (valence only)},\\
    \bm{\delta}, & k=2 \text{ (both targets)}.
  \end{cases}
\end{equation}
Three design choices keep the face stream from harming clips where it is unreliable:
\begin{enumerate}
  \item \emph{Zero initialisation.} The last layer of $g$ starts at zero, so training begins from the face-free prediction and the face stream learns only a correction.
  \item \emph{Gating to zero.} A clip with no gated step in any window receives $\bm{\Delta}_{\text{face}}=\mathbf{0}$ exactly, rather than the head's learned bias.
  \item \emph{Face dropout.} During training, a clip's entire face stream is hidden with probability $p_f$ by setting $m_{r,t}\leftarrow m_{r,t}(1-\xi)$, $\xi\sim\mathrm{Bernoulli}(p_f)$, for all $r,t$. This forces the rest of the network to remain predictive without the face.
\end{enumerate}

\subsection{Residual prediction on a context lookup}
\label{sec:residual}

A lookup of the typical label for each combination of event indicators is a strong predictor of the annotators' labels. Rather than make the network re-learn this lookup, ContextPose predicts a residual on top of it:
\begin{equation}
  \vy = \vy_{\text{pose}} + \bm{\Delta}_{\text{face}} + \vb_{g(\vc)}.
  \label{eq:output}
\end{equation}
Here $g(\vc)$ is the event-indicator combination of the clip and $\vb_{g}\in\R^2$ is a fixed offset. The offset is estimated only from the training clips, separately in each cross-validation fold, and is not trained by gradient descent. When the network is trained on annotator votes (\S\ref{sec:objective}), the offset is the log-odds of the group's mean vote share $\bar{\pi}_{g}$, clipped to $[\epsilon,1-\epsilon]$ with $\epsilon=0.02$:
\begin{equation}
  \vb_{g} = \log\frac{\bar{\pi}_{g}}{1-\bar{\pi}_{g}}.
\end{equation}
For regression it is the group's median label. Groups with fewer than three training clips use the overall value. Replacing $g(\vc)$ with a constant gives a variant in which no annotation information reaches the model through the offset.

\subsection{Output parameterisation and objective}
\label{sec:objective}

The model is trained on the annotators' votes. For each target, $\pi\in[0,1]$ is the share of annotators who voted positive valence (or high arousal). The output $\hat{y}$ is read as the log-odds that an annotator gives that vote. The loss is the binary cross-entropy against the soft vote share, summed over the two targets:
\begin{equation}
  \mathcal{L} = -\sum_{k\in\{\text{val},\,\text{aro}\}}\Big[\pi_k\log\sigma(\hat{y}_k)+(1-\pi_k)\log\big(1-\sigma(\hat{y}_k)\big)\Big].
\end{equation}
At inference, $\sigma(\hat{y}_k)$ is the score, and a binary call is made by thresholding it. In the regression setting, the same outputs are read directly as consensus valence and arousal and trained with a mean squared error.

\subsection{Stream ablations}
\label{sec:ablations}

The same network supports single-stream and partial-stream models, so each input's contribution can be measured with the architecture held fixed:
\begin{align}
  \text{pose only:}\quad & \vy = \vy_{\text{pose}}\big|_{\vchat=\mathbf{0}} + \bm{\Delta}_{\text{face}} + \vb, \\
  \text{context only:}\quad & \vy = \mathrm{MLP}\big(\Drop_{p_h}(\vchat)\big) + \bm{\Delta}_{\text{face}} + \vb, \\
  \text{face only:}\quad & \vy = \bm{\Delta}_{\text{face}} + \vb.
\end{align}
In the pose-only model the projected context is set to zero (for FiLM, the context input is zeroed). The query then reduces to the region embedding, so attention is no longer conditioned on the event. The face term is present only when the face stream is enabled. The offset $\vb$ is either the context lookup $\vb_{g(\vc)}$ or the constant.

\subsection{Implementation details}
\label{sec:impl}

Table~\ref{tab:hparams} lists the architectural hyper-parameters. The model is deliberately small, because the development set has only about 150 clips. Without the face stream, the annotation-queried model has about $74$k parameters. The face stream adds about $29$k, most of them in the $768\to32$ face projection.

\begin{table}[h]
\centering
\caption{Architectural hyper-parameters of ContextPose.}
\label{tab:hparams}
\small
\begin{tabular}{@{}llr@{}}
\toprule
Component & Hyper-parameter & Value \\
\midrule
Input & Steps per window $T$ / joints $J$ / features $F$ & 32 / 11 / 4 \\
& Context dimension $d_c$ (indicators / CLIP / VLM) & 4 / 768 / 384 \\
Pose encoder & Conv layers / kernel / activation & 2 / 3 / GELU \\
Model width & $W$ & 64 \\
Cross-attention & Heads $H$ / head width $d_h$ & 4 / 16 \\
Region Transformer & Layers / heads / FFN width & 1 / 4 / $2W$ \\
FiLM & Hidden width & 64 \\
Face stream & Input $D$ / encoding width & 768 / 32 \\
Head & Hidden width & $W/2$ \\
Regularisation & Transformer dropout / head dropout $p_h$ / face dropout $p_f$ & 0.3 / 0.3 / 0.3 \\
\midrule
Parameters & Without / with face stream (indicator context) & 73.6k / 102.9k \\
\bottomrule
\end{tabular}
\end{table}

\end{document}
</pasted_content id="3ab9">

Rewrite the methodology for the Context Pose Architecture. Reduce bullets and use paragraphs to make the info whole. You can make reference from the code.

---


**Claude**

Reading through the repo code now to check the methodology against the implementation before rewriting it.

---


**Claude**

Code checked: the paper's parameter counts (73.6k / 102.9k) are correct, and I found a few spots where the draft and the code disagree. Next I'll write the prose version.

---


**Claude**

I've rewritten the methodology in [reports/contextpose_methodology.tex](reports/contextpose_methodology.tex). There are no bullet lists left: the five-component overview, the context forms, the face quality gate and the face-stream safeguards are all paragraphs now. All your equations, the figure and the hyper-parameter table are kept. I couldn't compile it because LaTeX isn't installed on this machine. A script check of braces, environments and `\ref`s found no problems.

I checked the draft against `model.py`, `train.py`, `prepare.py`, `face.py` and `votes.py`. Your parameter counts are right: 73,634 without the face stream and 102,948 with it. These are the places where the draft and the code disagreed, or where the code had details worth adding:

- **The letter `g` meant two things.** It was the face head and also the lookup group in $\mathbf{b}_{g(\mathbf{c})}$. I renamed the face head to $\psi$, in the text and in the figure.
- **The inter-region Transformer doesn't know which summary is which window.** It has no positional encoding, and it ends in a mean, so swapping the three summaries gives the same output. The only place window identity enters is the region embedding, which shapes each window's attention. I toned down the "compare before, during and after" claim and said this explicitly.
- **There is no "constant offset" variant in the code.** Without `--residual-by`, the offset is zero and the head's bias does the job. The text now says that.
- **Speed handling has more detail.** If the previous frame is missing, the code uses the next one, and a joint with no observed neighbour gets zero speed. Speed is averaged over a centred window one resampling step wide.
- **Missing joints are now described.** They have zero coordinates and zero confidence, and resampling doesn't fill in the gaps.
- **The face gate thresholds are in the text:** YuNet score at least 0.8, brightness at least 40, sharpness at least 10, and a crop 0.9 shoulder widths wide.
- **CLIP sentences:** I added that they are generated from the indicators, so they carry the same information. I also added how the VLM embedding is built: Qwen3-Omni descriptions encoded with MiniLM.
- **How binary calls are made:** the threshold is the one that maximises balanced accuracy on the clips the model was fitted and selected on.
- **Dropout:** the Transformer also applies dropout to its attention weights.
- **New training paragraph and table rows.** These cover AdamW (learning rate 3e-4, weight decay 0.05), batch size 16, up to 60 epochs with patience 10, the augmentation, and the 5-seed ensemble. I also added measured parameter counts for the FiLM (82.3k), VLM (98.0k) and CLIP (122.5k) variants.

Three facts come from the README and the existing report rather than the code, so please confirm them: the 150-epoch cap for the VLM runs, the "nine distinct sentences" in the development clips, and Qwen3-Omni with all-MiniLM-L6-v2 for the VLM embeddings.

---


**User**

remove everything FilM from it

---


**Claude**

I've removed everything about FiLM from [reports/contextpose_methodology.tex](reports/contextpose_methodology.tex):

- **Subsection:** the whole "FiLM conditioning variant" subsection is gone, including its two equations, $f_c$, $\bm{\gamma}$ and $\bm{\beta}$, and the sentence comparing the two variants.
- **Figure caption:** I cut the last sentence, which described the FiLM variant.
- **Stream ablations:** I dropped the note "(for FiLM, the context input itself is zeroed)".
- **Implementation details:** I removed the 82.3k FiLM parameter count. "The attention-queried model" is now just "the model".
- **Table:** I deleted the "FiLM / Hidden width / 64" row.

Removing FiLM freed up the names $\mathbf{W}_1$ and $\mathbf{W}_2$. I renumbered the Transformer's feed-forward weights to $\mathbf{W}_1, \mathbf{W}_2$ and the prediction head's to $\mathbf{W}_3, \mathbf{W}_4$, so the numbering starts at 1 again.

A search finds no "FiLM" left in the file. Braces, environments and cross-references still check out. LaTeX isn't installed here, so I haven't compiled it.

---


**User**

remove these references too: \S\ref

---


**Claude**

All eight are in parentheses, so deleting them leaves every sentence grammatical. I'll remove them in one pass.

---


**Claude**

I removed all eight `\S\ref{...}` references from [reports/contextpose_methodology.tex](reports/contextpose_methodology.tex):

- **Overview paragraph:** six, one after each of the five components plus one after the closing sentence on the residual offset.
- **Problem formulation:** the one after "whose meaning depends on the training objective".
- **Residual section:** the one after "When the network is trained on annotator votes".

Every one was in parentheses, so deleting it left the sentence grammatical without any rewording.

I kept the other cross-references: `Figure~\ref{fig:arch}`, `Table~\ref{tab:hparams}`, and the two equation references (`eq:attn`, `eq:face`). Tell me if you want those gone as well. The section `\label`s are still in the file but nothing points to them now; they don't affect the output. Braces, environments and references still check out, but LaTeX isn't installed here, so I haven't compiled it.

---


**User**

what does this mean: The context therefore contributes no features of its own to the pose summary; it selects which time
steps of each window are summarised

---


**Claude**

It describes where the context enters the cross-attention, and what it is allowed to change there.

**How the summary is built.** For window $r$, the cross-attention output is

$$\mathbf{s}_r = \Big[\textstyle\sum_t \alpha_{r,1,t}\,\mathbf{v}_{r,1,t} \;\Vert\; \cdots \;\Vert\; \sum_t \alpha_{r,H,t}\,\mathbf{v}_{r,H,t}\Big]\mathbf{W}^O$$

Each part comes from a different source:

- **Values $\mathbf{v}_{r,i,t}$** come only from the pose tokens $\mathbf{Z}_r$. They are what actually gets added up.
- **Weights $\alpha_{r,i,t}$** come from comparing the query with the keys, and the query is $\hat{\mathbf{c}} + \mathbf{e}_r$. This is the only place the context appears.

So the numbers inside $\mathbf{s}_r$ are always a weighted average of pose-derived vectors. The context never gets copied into $\mathbf{s}_r$. All it does is set the weights, which decide how much each of the 32 time steps counts.

**An analogy.** Think of the window as 32 snapshots of the child's movement and the context as a spotlight. A social-interaction event might light up the steps just after the event starts. An auditory event might light up a different stretch. The summary describes what's under the light, and it is still made only of the child's movement. The spotlight picks where to look but adds nothing to the picture.

**The alternative it's contrasting with.** A common design averages the pose over time and then concatenates the context: $[\text{pose summary};\, \mathbf{c}]$. There, the context values sit directly in the vector the head reads. In ContextPose's attention they don't. That's the whole point of the design: the event decides which moments of behaviour matter, rather than being one more input feature.

**One caveat.** The sentence is strictly true for the window summary $\mathbf{s}_r$. The context can still reach the final output directly by three other routes:

- the optional context skip, $\mathbf{h} \leftarrow \mathbf{h} + \hat{\mathbf{c}}$;
- the context-lookup offset $\mathbf{b}_{g(\mathbf{c})}$;
- the context-only ablation.

If a reader might take the sentence as a claim about the whole network, a more explicit version would be:

> *Within the cross-attention, the context enters only through the query, so it changes only the attention weights: each window summary is a weighted average of pose features, and the context decides which time steps receive the weight.*

---


**User**

<pasted_content id="3ab9">
\documentclass[11pt]{article}


\usepackage[margin=2.3cm]{geometry}

\usepackage[T1]{fontenc}

\usepackage{lmodern}

\usepackage{amsmath,amssymb,bm}

\usepackage{booktabs}

\usepackage{xcolor}

\usepackage{tikz}

\usetikzlibrary{positioning,arrows.meta,calc}

\usepackage{caption}

\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}


\captionsetup{font=small,labelfont=bf}


\tikzset{
  box/.style={draw,rounded corners=2pt,align=center,font=\footnotesize,minimum height=7mm,inner sep=3pt,fill=white},
  inp/.style={box,fill=blue!6},
  op/.style={box,fill=gray!8},
  outputnode/.style={box,fill=green!8},
  arr/.style={-{Stealth[length=2mm]},thick},
}


% Notation

\newcommand{\R}{\mathbb{R}}

\newcommand{\regions}{\mathcal{R}}

\newcommand{\vc}{\mathbf{c}}

\newcommand{\vchat}{\hat{\mathbf{c}}}

\newcommand{\ve}{\mathbf{e}}

\newcommand{\vq}{\mathbf{q}}

\newcommand{\vs}{\mathbf{s}}

\newcommand{\vh}{\mathbf{h}}

\newcommand{\vy}{\hat{\mathbf{y}}}

\newcommand{\vb}{\mathbf{b}}

\newcommand{\vu}{\mathbf{u}}

\newcommand{\mZ}{\mathbf{Z}}

\newcommand{\mX}{\mathbf{X}}

\newcommand{\mS}{\mathbf{S}}

\newcommand{\mW}{\mathbf{W}}

\newcommand{\GELU}{\operatorname{GELU}}

\newcommand{\ReLU}{\operatorname{ReLU}}

\newcommand{\LN}{\operatorname{LN}}

\newcommand{\Drop}{\operatorname{Drop}}

\newcommand{\softmax}{\operatorname{softmax}}


\title{ContextPose}

\author{}

\date{}


\begin{document}

\maketitle


\section{Methodology}

\label{sec:method}


ContextPose estimates the valence and arousal reflected in human annotations of short video clips depicting a child before, during, and after an environmental event. The proposed architecture is motivated by the hypothesis that contextual information can help identify the behavioural moments most relevant to an affective response. Rather than merely concatenating context with an aggregated pose feature, we use the context representation as an attention query over the temporal pose features. In this formulation, context determines how the pose sequence is summarised, while the attended values themselves are derived from observed body movements.


The framework is illustrated in Figure~\ref{fig:arch}. Pose trajectories from the pre-event, event, and post-event intervals are first encoded independently using a shared temporal convolutional network. Event context, represented either through annotations or a language-based description, is mapped into the same latent space. For each interval, the context embedding is combined with a learned region embedding to query the corresponding temporal pose tokens using multi-head cross-attention. The resulting three summaries are integrated by a single-layer Transformer and passed to a prediction head for valence and arousal. When reliable facial observations are available, an auxiliary face branch provides an additive correction. The final estimate is formulated as a residual relative to a fixed baseline derived from the event-indicator configuration.


%-----------------------------------------------------------------------------

\begin{figure}[t]

\centering

\resizebox{\linewidth}{!}{%

\begin{tikzpicture}[every node/.style={text width=1.75cm}]

  % Pose columns

  \foreach \name/\x/\lbl in {pre/0/pre-event, ev/3/event, post/6/post-event}{

    \node[inp] (P\name) at (\x,0) {Pose $\mathbf{P}_{\text{\lbl}}$\\[-1pt]{\scriptsize $T{\times}J{\times}F$}};

    \node[op]  (T\name) at (\x,-1.4) {Temporal conv.\\[-1pt]{\scriptsize $\to \mZ_r\in\R^{T\times W}$}};

    \node[op,minimum height=11mm] (A\name) at (\x,-2.9) {Cross-attn.\\[-1pt]{\scriptsize $\vq_r=\vchat+\ve_r$}};

    \draw[arr] (P\name) -- (T\name);

    \draw[arr] (T\name) -- node[right,font=\scriptsize]{$\mZ_r$} (A\name);

  }


  % Context path

  \node[inp,text width=2.3cm] (C) at (-2.9,1.35) {Context $\vc$\\[-1pt]{\scriptsize indicators / CLIP / VLM}};

  \node[op] (Wc) at (-0.15,1.35) {Linear\\[-1pt]{\scriptsize $\vchat=\mW_c\vc+\vb_c$}};

  \draw[arr] (C) -- (Wc);

  \draw[thick] (Wc.east) -- (7.25,1.35);

  \foreach \name/\x in {pre/1.25, ev/4.25, post/7.25}{

    \draw[arr] (\x,1.35) |- (A\name.east);

  }


  % Region transformer and head

  \node[op,text width=7.6cm] (RT) at (3,-4.6) {Inter-region Transformer over $[\vs_{\text{pre}};\vs_{\text{event}};\vs_{\text{post}}]$, then mean over regions $\to \vh$};

  \foreach \name in {pre,ev,post}{ \draw[arr] (A\name) -- node[right,font=\scriptsize]{$\vs_r$} (A\name |- RT.north); }

  \node[op,text width=3.2cm] (H) at (3,-5.85) {MLP head $W\!\to\!W/2\!\to\!2$};

  \draw[arr] (RT) -- (H);

  \node[draw,circle,inner sep=1pt,text width=0.4cm,font=\small] (plus) at (3,-7.0) {$+$};

  \draw[arr] (H) -- node[right,font=\scriptsize]{$\vy_{\text{pose}}$} (plus);

  \node[inp,text width=2.6cm] (bg) at (-0.2,-7.0) {Offset $\vb_{g(\vc)}$\\[-1pt]{\scriptsize context lookup}};

  \draw[arr] (bg) -- (plus);

  \node[outputnode,text width=2.6cm] (y) at (3,-8.1) {$\vy=[\hat y_{\text{val}},\hat y_{\text{aro}}]$};

  \draw[arr] (plus) -- (y);


  % Face stream

  \node[inp,text width=2.4cm] (F) at (9.7,0) {Face features $\mathbf{x}^{f}_{r,t}$, gate $m_{r,t}$};

  \node[op,text width=2.4cm] (FE) at (9.7,-1.4) {LN $\to$ Linear $\to$ GELU\\[-1pt]{\scriptsize $\vu_{r,t}\in\R^{32}$}};

  \node[op,text width=2.4cm,minimum height=11mm] (FP) at (9.7,-2.9) {Gated mean per region\\[-1pt]{\scriptsize $\bar{\vu}_r$}};

  \node[op,text width=2.4cm] (FH) at (9.7,-4.6) {Face head $\psi$\\[-1pt]{\scriptsize zero-initialised}};

  \draw[arr] (F) -- (FE);

  \draw[arr] (FE) -- (FP);

  \draw[arr] (FP) -- (FH);

  \draw[arr] (FH.south) |- node[pos=0.75,above,font=\scriptsize]{$\bm{\Delta}_{\text{face}}$} (plus.east);

\end{tikzpicture}}

\caption{ContextPose. A temporal convolution encodes each window's pose sequence, and a query formed from the projected context and a learned region embedding attends over the window's time steps; the convolution and the cross-attention share their weights across the three windows. A Transformer layer relates the three window summaries, which are then averaged and decoded to valence and arousal. An optional gated face stream adds a zero-initialised correction, and the output is a residual on a context-lookup offset.}

\label{fig:arch}

\end{figure}

%-----------------------------------------------------------------------------


\subsection{Problem formulation and inputs}

\label{sec:inputs}


We organise each clip into three event-centred intervals, indexed by $r\in\regions=\{\text{pre},\text{event},\text{post}\}$: a 5\,s pre-event interval, the event interval itself, and a 5\,s post-event interval. Clips lasting 10\,s or less are omitted because they contain no event frames under this partition. The pose input for each interval is $\mathbf{P}_r\in\R^{T\times J\times F}$, while the overall clip is associated with an event-context vector $\vc\in\R^{d_c}$. If the facial branch is active, each interval additionally provides face embeddings $\mathbf{X}^f_r\in\R^{T\times D}$ and a binary validity mask $\mathbf{m}_r\in\{0,1\}^{T}$. The network outputs two unconstrained values $\vy\in\R^2$, corresponding to valence and arousal, whose interpretation is specified by the selected learning objective.


% \paragraph{Pose representation.}
 % \We retain $J=11$ upper-body COCO landmarks, comprising the nose, eyes, ears, shoulders, elbows, and wrists, extracted at $f_s=25$\,fps. Let $\mathbf{p}_{n,j}\in\R^2$ denote the image coordinates of joint $j$ in frame $n$. To reduce sensitivity to camera placement and apparent subject scale, coordinates are centred at the shoulder midpoint $\mathbf{o}_n$ and normalised by the clip-level median shoulder width $w$:

% \begin{equation}
%   \tilde{\mathbf{p}}_{n,j} = \frac{\mathbf{p}_{n,j}-\mathbf{o}_n}{w},
% \end{equation}

% This transformation suppresses differences in image position and apparent camera distance while retaining relative body configuration. To obtain fixed-size inputs from intervals of unequal duration, we resample each interval to $T=32$ temporal steps. Each joint is described by $F=4$ features: two normalised coordinates, detector confidence restricted to $[0,1]$, and movement speed. Missing detections are assigned zero coordinates and confidence. During interpolation, a step uses its two surrounding observations when both exist; if only one exists, that observation is retained, and if neither exists, the step remains zero. This avoids inventing trajectories across missing observations, with the confidence feature explicitly indicating uncertainty.


% Fixed-length resampling alone does not preserve movement rate: an event lasting 20\,s and a surrounding interval lasting 5\,s would both contain 32 steps despite their different physical durations. We therefore calculate joint speed from the original frame sequence, before temporal resampling, and express it in shoulder widths per second:

% \begin{equation}
%   v_{n,j} = \min\!\left(f_s\,\lVert\tilde{\mathbf{p}}_{n,j}-\tilde{\mathbf{p}}_{n-1,j}\rVert_2,\; v_{\max}\right), \qquad v_{\max}=20.
% \end{equation}

% If a valid previous observation is unavailable, the next observed frame is used; joints lacking a valid neighbouring observation receive zero speed. The limit $v_{\max}$ reduces sensitivity to abrupt tracking errors. For each resampled step, the speed feature is the average over observed frames in a centred span of approximately $N_r/T$ original frames, where $N_r$ is the interval length. Accordingly, the feature retains a consistent real-time interpretation across intervals.


\subsection{Temporal pose encoder}

\label{sec:pose-encoder}


For each region, we concatenate the joint-level features at every step to form $\mX_r\in\R^{T\times JF}$. Two one-dimensional temporal convolutional layers, each with kernel size 3 and zero padding, transform the sequence into $W$-dimensional temporal tokens:

\begin{equation}
  \mZ_r = \GELU\!\big(\mathrm{Conv}_2\big(\GELU(\mathrm{Conv}_1(\mX_r))\big)\big)\in\R^{T\times W},
  \qquad
  \big[\mathrm{Conv}(\mX)\big]_t = \vb + \sum_{k=-1}^{1}\mW_k\,\mathbf{x}_{t+k},
\end{equation}

Boundary inputs satisfy $\mathbf{x}_0=\mathbf{x}_{T+1}=\mathbf{0}$. The first convolution has weights $\mW_k\in\R^{W\times JF}$, whereas the second uses $\mW_k\in\R^{W\times W}$. Stacking these layers yields a five-step receptive field, enabling each token $\mathbf{z}_{r,t}$ to encode local motion rather than an isolated pose. Parameters are shared across the three regions. No explicit temporal positional embedding is introduced: ordering is represented locally through convolution, while the subsequent within-region attention is otherwise permutation-invariant over the generated tokens.


\subsection{Context encoder}

\label{sec:context}


We investigate three alternative representations of environmental context, each providing a different level of semantic detail. The first consists of four annotator-defined binary indicators describing social interaction, auditory events, activity or object changes, and visual or media events ($d_c=4$); clips without annotated events receive an all-zero vector. The second expresses these indicators as natural-language event descriptions and encodes the sentences using a frozen CLIP ViT-L/14 text encoder, followed by $\ell_2$ normalisation ($d_c=768$). As this text is generated from the event indicators, it does not add independent scene content; only nine unique sentences occur in the development clips. The third representation instead derives descriptions from the video itself. Qwen3-Omni generates accounts of the scene, the child's behaviour, and audible surroundings over successive 5--6\,s segments. These descriptions are encoded using all-MiniLM-L6-v2, averaged across segments, and $\ell_2$-normalised to obtain a clip-level vector ($d_c=384$). Each representation is mapped to the common latent width using a linear projection:

\begin{equation}
  \vchat = \mW_c\,\vc + \vb_c\in\R^{W}.
\end{equation}

In addition, each temporal region is assigned a learned embedding $\ve_r\in\R^W$, with parameters initialised from $\mathcal{N}(0,0.02^2)$. Because the cross-attention module shares parameters across regions, these embeddings provide the region-specific identity needed to distinguish pre-event, event, and post-event queries.


\subsection{Context-queried temporal attention}

\label{sec:cross-attention}


The central fusion mechanism is context-conditioned temporal cross-attention. For each region, the model constructs a query from the projected environmental context and the corresponding region embedding; keys and values come exclusively from that region's pose tokens. The resulting single query vector is defined as:

\begin{equation}
  \vq_r = \vchat + \ve_r\in\R^{1\times W}.
\end{equation}

We apply multi-head attention with $H=4$ heads, each of dimension $d_h=W/H$. For head $i$, the query, key, value, and attention weights are computed as follows:

\begin{align}
  \mathbf{Q}_{r,i} &= \vq_r\mW^{Q}_i, \qquad
  \mathbf{K}_{r,i} = \mZ_r\mW^{K}_i, \qquad
  \mathbf{V}_{r,i} = \mZ_r\mW^{V}_i, \\
  \bm{\alpha}_{r,i} &= \softmax\!\left(\frac{\mathbf{Q}_{r,i}\mathbf{K}_{r,i}^{\top}}{\sqrt{d_h}}\right)\in\R^{1\times T}. \label{eq:attn}
\end{align}

The head outputs are concatenated and linearly projected into the region-level representation:

\begin{equation}
  \vs_r = \big[\bm{\alpha}_{r,1}\mathbf{V}_{r,1}\;\Vert\;\cdots\;\Vert\;\bm{\alpha}_{r,H}\mathbf{V}_{r,H}\big]\,\mW^{O}\in\R^{W},
\end{equation}

Here, $\mW^{Q}_i,\mW^{K}_i,\mW^{V}_i\in\R^{W\times d_h}$ and $\mW^{O}\in\R^{W\times W}$, with bias terms excluded from the notation. The same attention projections are reused across all three regions.


As shown by Equation~\eqref{eq:attn}, environmental context influences the attention distribution rather than being inserted directly into the pose-derived values. For each head, the region summary is a weighted average of projected pose tokens, and the context vector $\vc$ affects which tokens receive the greatest weights. Consequently, an auditory event and a social interaction may lead the model to emphasise different periods of the same behavioural sequence. This mechanism produces a fixed-dimensional summary $\vs_r$ for every region, independent of its original temporal extent.


\subsection{Inter-region Transformer and prediction head}

\label{sec:region-transformer}


The regional summaries are assembled into $\mS=[\vs_{\text{pre}};\vs_{\text{event}};\vs_{\text{post}}]\in\R^{3\times W}$. A single post-normalisation Transformer encoder with $H=4$ attention heads models relationships among these representations:

\begin{align}
  \mS' &= \LN\!\big(\mS + \Drop(\mathrm{MHSA}(\mS))\big), \\
  \mS'' &= \LN\!\big(\mS' + \Drop(\mathrm{FFN}(\mS'))\big), \qquad
  \mathrm{FFN}(\mathbf{x}) = \mW_2\,\Drop\!\big(\ReLU(\mW_1\mathbf{x}+\vb_1)\big)+\vb_2.
\end{align}

The operation $\mathrm{MHSA}$ denotes self-attention over the three region vectors; the feed-forward sublayer has width $2W$, and attention-weight dropout is applied. Through this operation, the representation of a given region can incorporate information from the others. For example, behaviour during the event can be interpreted in relation to the child's movements immediately before and afterward. We obtain a clip-level representation by averaging the transformed regional outputs:

\begin{equation}
  \vh = \frac{1}{3}\sum_{r\in\regions}\mS''_r\in\R^{W}.
\end{equation}

We do not apply positional encodings at this stage. Since self-attention followed by mean aggregation is invariant to permutations of the three input rows, the ordering of the regional vectors is not explicitly available to the final representation. The learned region embeddings introduced at the preceding attention stage are therefore the only source of region identity.


An optional \emph{context skip} connection modifies the fused representation as $\vh\leftarrow\vh+\vchat$, allowing context to influence the output directly as well as through temporal selection. A two-layer MLP, with input dropout and an intermediate dimension of $W/2$, maps the resulting representation to the pose-branch valence and arousal scores:

\begin{equation}
  \vy_{\text{pose}} = \mW_4\,\GELU\!\big(\mW_3\,\Drop_{p_h}(\vh)+\vb_3\big)+\vb_4\in\R^{2}.
\end{equation}


\subsection{Gated face stream}

\label{sec:face}


Although facial expression can provide strong evidence of valence, its reliability varies considerably in naturalistic recordings of children. Face crops may be occluded, blurred, poorly illuminated, or affected by head rotation. We therefore model facial information through a separate auxiliary branch that contributes an additive correction to the pose-based estimate only when adequate facial observations are available.


For each of the $T$ temporal positions in a region, we extract the face crop from the nearest video frame. The crop is centred using the tracked facial landmarks, covers a square region of width 0.9 shoulder widths, and is resized to 224\,px. A frozen facial-expression ViT pretrained on FER2013 produces a $D=768$-dimensional [CLS] embedding $\mathbf{x}^f_{r,t}\in\R^{D}$. A binary mask $m_{r,t}$ identifies valid crops according to three criteria: (i) the pose tracker detects the nose and both eyes, indicating a frontal face; (ii) YuNet independently detects a face at that location with confidence at least 0.8; and (iii) the crop exceeds a mean grey-level threshold of 40 and a Laplacian-variance threshold of 10. These checks reject unreliable detections, including cases in which the pose estimator reports facial landmarks despite occlusion or a lowered head. Invalid steps have their features set to zero.


We first project each face embedding through layer normalisation, a linear transformation, and GELU activation:

\begin{equation}
  \vu_{r,t} = \GELU\!\big(\mW_f\,\LN(\mathbf{x}^f_{r,t})+\vb_f\big)\in\R^{32},
\end{equation}

The resulting step features are aggregated using a mask-weighted mean within each region:

\begin{equation}
  \bar{\vu}_r = \frac{\sum_{t=1}^{T} m_{r,t}\,\vu_{r,t}}{\max\!\big(1,\sum_{t=1}^{T} m_{r,t}\big)},
\end{equation}

The denominator $\max(1,\cdot)$ ensures that a region with no valid facial frames has a zero summary. The three 32-dimensional summaries are concatenated and passed to a two-layer prediction head $\psi$ ($96\to W/2\to k$), generating the following correction:

\begin{equation}
  \bm{\delta} = \psi\big(\Drop_{p_h}([\bar{\vu}_{\text{pre}};\bar{\vu}_{\text{event}};\bar{\vu}_{\text{post}}])\big)\in\R^{k},
  \qquad
  \bm{\Delta}_{\text{face}} = \mathbb{I}\Big[\textstyle\sum_{r,t} m_{r,t}>0\Big]\cdot
  \begin{cases}
    [\delta_1,\,0]^{\top}, & k=1 \text{ (valence only)},\\
    \bm{\delta}, & k=2 \text{ (both targets)}.
  \end{cases}
  \label{eq:face}
\end{equation}


We introduce three safeguards to limit the effect of low-quality facial information. First, the final layer of $\psi$ is initialised to zero, so the model initially reproduces the pose-and-context prediction and only later learns an additional facial contribution. Second, the indicator in Equation~\eqref{eq:face} forces $\bm{\Delta}_{\text{face}}=\mathbf{0}$ when no valid facial step exists anywhere in the clip, preventing a spurious correction from the head's bias. Third, during training, the complete face branch is randomly removed with probability $p_f$ by applying $m_{r,t}\leftarrow m_{r,t}(1-\xi)$ for all regions and steps, where $\xi\sim\mathrm{Bernoulli}(p_f)$. This face dropout also makes the correction zero and encourages the remaining modalities to support the prediction independently.


\subsection{Residual prediction on a context lookup}

\label{sec:residual}


Because event-indicator combinations are informative about the observed annotation distribution, a simple context-conditioned lookup provides a useful baseline. Given the small development sample of approximately 150 clips, we train ContextPose to estimate deviations from this baseline rather than requiring its neural branches to learn the full prediction directly:

\begin{equation}
  \vy = \vy_{\text{pose}} + \bm{\Delta}_{\text{face}} + \vb_{g(\vc)}.
  \label{eq:output}
\end{equation}

In Equation~\eqref{eq:output}, $g(\vc)$ identifies the event-indicator combination associated with the clip, and $\vb_{g}\in\R^2$ denotes its fixed baseline offset. Offsets are estimated exclusively from training examples within each cross-validation fold and are not updated by gradient descent, thereby excluding evaluation data from baseline estimation. For vote-based learning, each offset equals the log-odds of the group's mean vote proportion $\bar{\pi}_{g}$. We clip this proportion to $[\epsilon,1-\epsilon]$, where $\epsilon=0.02$, to avoid infinite log-odds:

\begin{equation}
  \vb_{g} = \log\frac{\bar{\pi}_{g}}{1-\bar{\pi}_{g}}.
\end{equation}

For regression, the offset is instead the median target value for the corresponding group. If fewer than three training clips are available for a group, the global training-set estimate is used. Disabling the lookup sets the offset to zero; the prediction head's bias then supplies any learned constant component, and no label statistics are injected through this branch.


\subsection{Output parameterisation and objective}

\label{sec:objective}


Three annotators assessed each clip, assigning positive, neutral, or negative valence and either high or low arousal. Given the moderate inter-annotator agreement, our primary objective retains the distribution of votes instead of collapsing them to a single consensus class. For each dimension, $\pi\in[0,1]$ denotes the proportion of known votes for positive valence or high arousal. The prediction $\hat{y}$ is interpreted as the logit of that vote probability, and training minimises binary cross-entropy with the soft vote proportion for both outputs, averaged over the batch:

\begin{equation}
  \mathcal{L} = -\sum_{k\in\{\text{val},\,\text{aro}\}}\Big[\pi_k\log\sigma(\hat{y}_k)+(1-\pi_k)\log\big(1-\sigma(\hat{y}_k)\big)\Big].
\end{equation}

During inference, $\sigma(\hat{y}_k)$ gives the estimated vote probability, used to distinguish positive from non-positive valence or high from low arousal after applying a threshold. Thresholds are chosen to maximise balanced accuracy using only model-fitting and selection data, never evaluation clips. For the alternative regression objective, the two outputs represent consensus valence and arousal directly and are optimised using mean squared error.


\subsection{Stream ablations}

\label{sec:ablations}


To assess the contribution of each modality, we evaluate stream-removal variants that retain the same basic prediction framework while disabling selected inputs:

\begin{align}
  \text{pose only:}\quad & \vy = \vy_{\text{pose}}\big|_{\vchat=\mathbf{0}} + \bm{\Delta}_{\text{face}} + \vb, \\
  \text{context only:}\quad & \vy = \mathrm{MLP}\big(\Drop_{p_h}(\vchat)\big) + \bm{\Delta}_{\text{face}} + \vb, \\
  \text{face only:}\quad & \vy = \bm{\Delta}_{\text{face}} + \vb.
\end{align}

In the pose-only variant, the projected context is set to zero, leaving the learned region embeddings to provide the cross-attention queries. In the context-only variant, the pose pathway is bypassed and the prediction MLP operates directly on the projected context. The face-only configuration removes the pose prediction and retains the facial correction where that branch is enabled. Across these experiments, $\vb$ denotes either the fixed context-dependent offset $\vb_{g(\vc)}$ or zero when the lookup is inactive. Thus, when the lookup is enabled, the nominal face-only condition still incorporates event information through its baseline offset.


\subsection{Implementation details}

\label{sec:impl}


We keep the network compact to limit overfitting on the development set of approximately 150 clips. Table~\ref{tab:hparams} summarises the architectural and optimisation settings. With latent width $W=64$, the indicator-context configuration contains 73.6k trainable parameters; using VLM or CLIP features increases this to 98.0k and 122.5k, respectively, primarily due to the input projection. The optional face branch introduces another 29.3k parameters, largely from mapping the 768-dimensional face embedding to 32 dimensions.


We optimise all variants using AdamW with learning rate $3\times10^{-4}$, weight decay 0.05, and batch size 16. Training lasts at most 60 epochs for indicator and CLIP contexts or 150 epochs for VLM context, with early stopping after 10 epochs without validation improvement. The selected checkpoint minimises validation cross-entropy in the vote-prediction setting and validation mean absolute error in regression. To augment each training sequence, we apply random rotations within $\pm15^\circ$, scale factors in $[0.9,1.1]$, and Gaussian coordinate noise of standard deviation $\sigma=0.02$ shoulder widths. Horizontal reflection is applied with probability 0.5, reversing the horizontal coordinate and exchanging left and right joints. All regions of a clip receive the same transform, and missing joints remain unchanged. For final evaluation, we ensemble five independently seeded models, averaging their prediction scores and using a majority vote of their binary decisions.


\begin{table}[h]

\centering

\caption{Hyper-parameters of ContextPose.}

\label{tab:hparams}

\small

\begin{tabular}{@{}llr@{}}
\toprule
Component & Hyper-parameter & Value \\
\midrule
Input & Steps per window $T$ / joints $J$ / features $F$ & 32 / 11 / 4 \\
& Context dimension $d_c$ (indicators / CLIP / VLM) & 4 / 768 / 384 \\
Pose encoder & Conv layers / kernel / activation & 2 / 3 / GELU \\
Model width & $W$ & 64 \\
Cross-attention & Heads $H$ / head width $d_h$ & 4 / 16 \\
Region Transformer & Layers / heads / FFN width & 1 / 4 / $2W$ \\
Face stream & Input $D$ / encoding width & 768 / 32 \\
Head & Hidden width & $W/2$ \\
Regularisation & Transformer dropout / head dropout $p_h$ / face dropout $p_f$ & 0.3 / 0.3 / 0.3 \\
\midrule
Optimisation & Optimiser / learning rate / weight decay & AdamW / $3{\times}10^{-4}$ / 0.05 \\
& Batch size / max.\ epochs / early-stopping patience & 16 / 60 / 10 \\
& Ensemble size (seeds) & 5 \\
\midrule
Parameters & Without / with face stream (indicator context) & 73.6k / 102.9k \\
\bottomrule
\end{tabular}

\end{table}


\end{document}
</pasted_content id="3ab9">

This is what I'm currently working with. For parts where we describe joint features as an input stream, joints features can also be replace by subject aware encoder directly from input videos such as EmotionClip

---


**User**

This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

<system-reminder>
<tools-used-before-summary>Bash, Edit, Read, Write, mcp__ccd_turn__update_plan, mcp__ccd_turn__wrap_up</tools-used-before-summary>
</system-reminder>
Summary:
1. Primary Request and Intent:
   - **Initial request:** rewrite the Methodology section of the user's pasted ContextPose LaTeX document. The user's words: "Reduce bullets and use paragraphs to make the info whole. You can make reference from the code". That means using the repo code to make the details accurate.
   - **Follow-up 1:** "remove everything FilM from it".
   - **Follow-up 2:** "remove these references too: \S\ref".
   - **Current question:** "what does this mean: The context therefore contributes no features of its own to the pose summary; it selects which time steps of each window are summarised". The user wants a conceptual explanation of that sentence, which sits in the first intro paragraph of the methodology.

2. Key Technical Concepts:
   - **Pose input.** Each clip is split into 3 windows (pre, event, post). There are 5 s flanks, and clips of 10 s or less are excluded. There are 11 COCO upper-body joints at 25 fps. Coordinates are normalised by the shoulder midpoint and the median shoulder width. Each window is resampled to T=32 steps. There are F=4 features per joint: x, y, confidence clipped to [0,1], and speed.
   - **Speed channel.** Speed is computed on the original frames. It falls back to the next frame if the previous one is missing, and is zero if no neighbour is observed. It is capped at 20 shoulder widths/s and averaged with a centred moving mean one resampling interval wide.
   - **Temporal encoder.** A 2-layer Conv1d (kernel 3, GELU), giving a receptive field of 5 steps. It is shared across windows and has no positional encoding.
   - **Context forms.**
     - 4 binary indicators (d_c=4).
     - CLIP ViT-L/14 text embedding (768). The sentences are generated from the indicators; there are only 9 distinct sentences among the development clips.
     - VLM vector (384): Qwen3-Omni descriptions of 5–6 s segments, encoded with all-MiniLM-L6-v2, averaged and L2-normalised.
     - In every case the context goes through one linear projection, and there is a learned region embedding e_r ~ N(0, 0.02²).
   - **Context-queried cross-attention.** The query is q_r = ĉ + e_r. Keys and values are the pose tokens Z_r. There are 4 heads (d_h=16 at W=64), with weights shared across windows. The summary s_r is a convex combination of the value-projected pose tokens. The context affects only the attention weights α.
   - **Inter-region Transformer.**
     - One post-norm layer: ReLU, FFN width 2W, and dropout including on the attention weights.
     - It is followed by a mean over the 3 regions.
     - It is order-invariant because there is no positional encoding and the output is averaged.
     - Region identity enters only through e_r's effect on attention.
     - There is an optional context skip, h ← h + ĉ.
     - The MLP head is W → W/2 → 2, with GELU and dropout p_h on its input.
   - **Gated face stream.**
     - Features: FER2013 ViT [CLS] (768-d) from a crop 0.9 shoulder widths wide, resized to 224 px, at the frame nearest each resampling step.
     - Quality gate: frontal (nose and both eyes tracked), YuNet score ≥ 0.8, grey-level brightness ≥ 40, Laplacian-variance sharpness ≥ 10.
     - Encoder: LN → Linear(768, 32) → GELU, then a masked mean per window.
     - Head: ψ (96 → W/2 → k), zero-initialised. It outputs exactly zero when no step is kept, and face dropout p_f is used in training. k=1 corrects valence only; k=2 corrects both targets.
   - **Residual offset.**
     - b_g(c) is estimated from the training clips only, separately in each fold.
     - For votes it is the log-odds of the group's mean vote share, clipped to [0.02, 0.98]. For regression it is the group's median label.
     - Groups with fewer than 3 clips use the overall value. With no lookup, the offset is zero.
     - In code this is `--residual-by event_text`, which is equivalent to the indicator combination.
   - **Votes objective.** The loss is BCE against the soft vote share, averaged over the batch and summed over the 2 targets. The binary-call threshold maximises balanced accuracy on the clips the model was fitted and selected on. In the ensemble, scores are averaged and the call is the members' majority. The regression alternative uses MSE.
   - **Training.**
     - AdamW, learning rate 3e-4, weight decay 0.05, batch 16.
     - Up to 60 epochs (150 for VLM), patience 10.
     - Augmentation: rotation ±15°, scale 0.9–1.1, jitter σ=0.02, and a mirror with p=0.5. The same transform is used for all 3 windows, and only observed joints are moved.
     - Final predictions come from a 5-seed ensemble.
   - **Parameter counts at W=64, J=11, F=4** (verified by running the code):

     | Model | Parameters |
     |---|---|
     | annotation_attention | 73,634 |
     | + face, both targets | 102,948 |
     | + face, valence only | 102,915 |
     | CLIP context | 122,530 |
     | VLM context | 97,954 |
     | FiLM (now removed from the doc) | 82,274 |

3. Files and Code Sections:
   - **/Users/henryansah/workspace/cvpr/reports/contextpose_methodology.tex** (CREATED, then edited twice). This is the deliverable.
     - **Structure:**
       - Section Methodology with 2 intro prose paragraphs.
       - TikZ figure (face head renamed g → ψ; FiLM caption sentence removed).
       - Subsections: Problem formulation and inputs (with a "Pose representation" paragraph and a speed paragraph), Temporal pose encoder, Context encoder, Context-queried temporal attention, Inter-region Transformer and prediction head, Gated face stream, Residual prediction on a context lookup, Output parameterisation and objective, Stream ablations, and Implementation details (with a training paragraph).
       - Table tab:hparams. It has no FiLM row and does have optimisation rows: AdamW / 3e-4 / 0.05; 16 / 60 / 10; ensemble 5. The parameters row reads 73.6k / 102.9k.
     - **Lists:** no itemize/enumerate remain.
     - **References:** no `\S\ref` remain. The remaining refs are `Figure~\ref{fig:arch}`, `Table~\ref{tab:hparams}`, `\eqref{eq:attn}` and `\eqref{eq:face}`.
     - **Current first intro paragraph** (contains the sentence the user asked about):
       "ContextPose predicts the valence and arousal that annotators assign to a short clip of a child recorded around an environmental event. Its design follows from a single premise: \emph{the event context should decide which moments of the child's behaviour matter}. Rather than appending the context to a pooled pose representation, ContextPose turns the context into the query of an attention operation over time. The context therefore contributes no features of its own to the pose summary; it selects which time steps of each window are summarised."
     - **Related paragraph in the attention subsection:** "Equation~\eqref{eq:attn} makes the role of the context explicit. Each summary $\vs_r$ is a convex combination of the projected step tokens of window $r$, and only the weights $\bm{\alpha}_{r,i}$ depend on $\vc$. The context thus decides \emph{which time steps} of the window are summarised, without adding information to the pose features. A social-interaction event, for example, can direct attention to a different part of the post-event window than an auditory event does. ..."
     - **Current equations after renumbering:**
       - FFN: `\mathrm{FFN}(\mathbf{x}) = \mW_2\,\Drop\!\big(\ReLU(\mW_1\mathbf{x}+\vb_1)\big)+\vb_2`
       - Head: `\vy_{\text{pose}} = \mW_4\,\GELU\!\big(\mW_3\,\Drop_{p_h}(\vh)+\vb_3\big)+\vb_4\in\R^{2}`
     - **Ablation text:** "In the pose-only model the projected context is replaced by zeros. The query then reduces to the region embedding..."
     - **Implementation text:** "With $W=64$ and the indicator context, the model has 73.6k parameters. A VLM or CLIP context raises this to 98.0k or 122.5k, almost entirely through the wider context projection. The face stream adds 29.3k parameters..."
     - **Must re-read before editing:** the file was last modified by Bash sed, so it has to be read again before any further use of Edit.
   - **src/asd_contextpose/model.py** (read). The key forward logic behind the user's question:
     ```python
     text = self.text_projection(condition)
     ...
     query = (text + self.region_embeddings[region]).unsqueeze(1)
     summary, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
     ...
     fused = self.temporal_encoder(torch.stack(regions, dim=1)).mean(dim=1)
     if c.text_skip:
         fused = fused + text
     return self.head(self.dropout(fused)) + face
     ```
     `self.cross_attention = nn.MultiheadAttention(width, 4, batch_first=True)`. It has no residual path from query to output.
   - **Other files read for verification:**
     - src/asd_contextpose/train.py: group_logits, MIN_RATE=0.02, WithOffset, augment, the BCE loss.
     - prepare.py: resample, joint_speed, bin_mean, face gate constants MIN_FACE_DETECTOR=0.8, MIN_FACE_BRIGHTNESS=40, MIN_FACE_SHARPNESS=10.
     - data.py.
     - votes.py: fit_calls, best_threshold (balanced accuracy).
     - crossval.py.
     - face.py: FACE_SIDE=0.9, CROP=224, YuNet.
     - README.md: the V3 config.
     - reports/asd_contextpose_report.tex (grepped): VLM details and the "9 distinct sentences" fact.

4. Errors and fixes:
   - **zsh glob:** `grep --include=*.py` gave "no matches found". Fixed by grepping explicit paths (src, tests, reports, README.md).
   - **No LaTeX compiler installed.** Used a Python static check instead (brace depth, begin/end balance, undefined refs) and told the user the file was not compiled.
   - **Wording error:** "share of an annotator's known votes" was corrected to "share of the annotators' known votes".
   - **Corrections to the user's original draft, made using the code:**
     - The name `g` was used both for the face head and for the lookup group; the face head is now ψ.
     - The region-order invariance of the Transformer plus mean is now stated.
     - The "constant offset" variant does not exist in code; it is now described as a zero offset.
     - Speed fallback details added.
     - Threshold selection described.
     - Dropout on the attention weights added.
     - Training details added.

5. Problem Solving:
   - The methodology was rewritten into connected prose and checked against the implementation.
   - FiLM was removed entirely, with the weights renumbered.
   - All `\S\ref` section references were removed; the sentences still read correctly because every reference was parenthetical.
   - The only open item is the user's conceptual question.

6. All user messages:
   - [Pasted the LaTeX document] "Rewrite the methodology for the Context Pose Architecture. Reduce bullets and use paragraphs to make the info whole. You can make reference from the code."
   - "remove everything FilM from it"
   - "remove these references too: \S\ref"
   - "what does this mean: The context therefore contributes no features of its own to the pose summary; it selects which time steps of each window are summarised"
   - No security constraints were stated by the user. The system context says the user's email and personal details must stay private and must not be sent to external services.

7. Pending Tasks:
   - Answer the user's question explaining what the sentence means. No file edits were requested.

8. Current Work:
   - The user asked: "what does this mean: The context therefore contributes no features of its own to the pose summary; it selects which time steps of each window are summarised". The answer should explain the following.
   - **What happens in the cross-attention:**
     - The query is q_r = ĉ + e_r (projected context plus region embedding).
     - The keys and values come only from the window's pose tokens Z_r.
     - The window summary is s_r = Σ_t α_t (z_t W^V) W^O: a weighted average, with softmax weights summing to 1, of transformed pose tokens.
   - **Why "no features of its own":**
     - The context changes only the weights α_t, i.e. how much each of the 32 time steps counts.
     - No component of the context vector is copied or added into s_r. nn.MultiheadAttention has no query residual.
     - So the summary is built entirely from pose information. The context acts like a spotlight or selector, not as extra input features.
     - Contrast this with concatenation, [pooled pose; c], where the context values themselves enter the representation.
   - **Example:** for a social-interaction event the attention might weight later post-event steps (the child's reaction), while for an auditory event it might weight steps right after onset. The same pose sequence gives different summaries depending on the event.
   - **Caveats:** the context can still reach the prediction directly in other parts of the model:
     - the optional context skip (h ← h + ĉ);
     - the context-only ablation;
     - the residual lookup offset b_g(c).
     The sentence describes only the pose-summary (attention) step.

9. Optional Next Step:
   - Answer the user's question in plain prose, as described in Current Work. No tools are needed.
   - Optionally, offer a clearer rewording of the sentence for the paper, for example: "The context does not add its own features to the pose summary; it only decides how much weight each time step of the window receives." Do not edit the file unless the user asks.

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: /Users/henryansah/.claude/projects/-Users-henryansah-workspace-cvpr/67cd9486-2e25-4f58-9bb8-bc406f97875a.jsonl
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.

The messages after this summary are the most recent messages from before compaction, kept verbatim. The summary was written without seeing them, so something it says has not happened yet may already have happened in them.

---


**User**

[Request interrupted by user]

---


**User**

<pasted_content id="3ab9">
\documentclass[11pt]{article}
\usepackage[margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc}
\usepackage{caption}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\captionsetup{font=small,labelfont=bf}
\tikzset{
box/.style={draw,rounded corners=2pt,align=center,font=\footnotesize,minimum height=7mm,inner sep=3pt,fill=white},
inp/.style={box,fill=blue!6},
op/.style={box,fill=gray!8},
outputnode/.style={box,fill=green!8},
arr/.style={-{Stealth[length=2mm]},thick},
}
% Notation
\newcommand{\R}{\mathbb{R}}
\newcommand{\regions}{\mathcal{R}}
\newcommand{\vc}{\mathbf{c}}
\newcommand{\vchat}{\hat{\mathbf{c}}}
\newcommand{\ve}{\mathbf{e}}
\newcommand{\vq}{\mathbf{q}}
\newcommand{\vs}{\mathbf{s}}
\newcommand{\vh}{\mathbf{h}}
\newcommand{\vy}{\hat{\mathbf{y}}}
\newcommand{\vb}{\mathbf{b}}
\newcommand{\vu}{\mathbf{u}}
\newcommand{\mZ}{\mathbf{Z}}
\newcommand{\mX}{\mathbf{X}}
\newcommand{\mS}{\mathbf{S}}
\newcommand{\mW}{\mathbf{W}}
\newcommand{\GELU}{\operatorname{GELU}}
\newcommand{\ReLU}{\operatorname{ReLU}}
\newcommand{\LN}{\operatorname{LN}}
\newcommand{\Drop}{\operatorname{Drop}}
\newcommand{\softmax}{\operatorname{softmax}}
\title{ContextPose}
\author{}
\date{}
\begin{document}
\maketitle
\section{Methodology}
\label{sec:method}
ContextPose estimates the valence and arousal reflected in human annotations of short video clips depicting a child before, during, and after an environmental event. The proposed architecture is motivated by the hypothesis that contextual information can help identify the behavioural moments most relevant to an affective response. Rather than merely concatenating context with an aggregated pose feature, we use the context representation as an attention query over the temporal pose features. In this formulation, context determines how the pose sequence is summarised, while the attended values themselves are derived from observed body movements.
The framework is illustrated in Figure~\ref{fig:arch}. Pose trajectories from the pre-event, event, and post-event intervals are first encoded independently using a shared temporal convolutional network. Event context, represented either through annotations or a language-based description, is mapped into the same latent space. For each interval, the context embedding is combined with a learned region embedding to query the corresponding temporal pose tokens using multi-head cross-attention. The resulting three summaries are integrated by a single-layer Transformer and passed to a prediction head for valence and arousal. When reliable facial observations are available, an auxiliary face branch provides an additive correction. The final estimate is formulated as a residual relative to a fixed baseline derived from the event-indicator configuration.
%-----------------------------------------------------------------------------
\begin{figure}[t]
\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[every node/.style={text width=1.75cm}]
% Pose columns
\foreach \name/\x/\lbl in {pre/0/pre-event, ev/3/event, post/6/post-event}{

```
\node[inp] (P\name) at (\x,0) {Pose $\mathbf{P}_{\text{\lbl}}$\\[-1pt]{\scriptsize $T{\times}J{\times}F$}};

\node[op]  (T\name) at (\x,-1.4) {Temporal conv.\\[-1pt]{\scriptsize $\to \mZ_r\in\R^{T\times W}$}};

\node[op,minimum height=11mm] (A\name) at (\x,-2.9) {Cross-attn.\\[-1pt]{\scriptsize $\vq_r=\vchat+\ve_r$}};

\draw[arr] (P\name) -- (T\name);

\draw[arr] (T\name) -- node[right,font=\scriptsize]{$\mZ_r$} (A\name);

```

}
% Context path
\node[inp,text width=2.3cm] (C) at (-2.9,1.35) {Context $\vc$\[-1pt]{\scriptsize indicators / CLIP / VLM}};
\node[op] (Wc) at (-0.15,1.35) {Linear\[-1pt]{\scriptsize $\vchat=\mW_c\vc+\vb_c$}};
\draw[arr] (C) -- (Wc);
\draw[thick] (Wc.east) -- (7.25,1.35);
\foreach \name/\x in {pre/1.25, ev/4.25, post/7.25}{

```
\draw[arr] (\x,1.35) |- (A\name.east);

```

}
% Region transformer and head
\node[op,text width=7.6cm] (RT) at (3,-4.6) {Inter-region Transformer over $[\vs_{\text{pre}};\vs_{\text{event}};\vs_{\text{post}}]$, then mean over regions $\to \vh$};
\foreach \name in {pre,ev,post}{ \draw[arr] (A\name) -- node[right,font=\scriptsize]{$\vs_r$} (A\name |- RT.north); }
\node[op,text width=3.2cm] (H) at (3,-5.85) {MLP head $W!\to!W/2!\to!2$};
\draw[arr] (RT) -- (H);
\node[draw,circle,inner sep=1pt,text width=0.4cm,font=\small] (plus) at (3,-7.0) {$+$};
\draw[arr] (H) -- node[right,font=\scriptsize]{$\vy_{\text{pose}}$} (plus);
\node[inp,text width=2.6cm] (bg) at (-0.2,-7.0) {Offset $\vb_{g(\vc)}$\[-1pt]{\scriptsize context lookup}};
\draw[arr] (bg) -- (plus);
\node[outputnode,text width=2.6cm] (y) at (3,-8.1) {$\vy=[\hat y_{\text{val}},\hat y_{\text{aro}}]$};
\draw[arr] (plus) -- (y);
% Face stream
\node[inp,text width=2.4cm] (F) at (9.7,0) {Face features $\mathbf{x}^{f}{r,t}$, gate $m{r,t}$};
\node[op,text width=2.4cm] (FE) at (9.7,-1.4) {LN $\to$ Linear $\to$ GELU\[-1pt]{\scriptsize $\vu_{r,t}\in\R^{32}$}};
\node[op,text width=2.4cm,minimum height=11mm] (FP) at (9.7,-2.9) {Gated mean per region\[-1pt]{\scriptsize $\bar{\vu}_r$}};
\node[op,text width=2.4cm] (FH) at (9.7,-4.6) {Face head $\psi$\[-1pt]{\scriptsize zero-initialised}};
\draw[arr] (F) -- (FE);
\draw[arr] (FE) -- (FP);
\draw[arr] (FP) -- (FH);
\draw[arr] (FH.south) |- node[pos=0.75,above,font=\scriptsize]{$\bm{\Delta}_{\text{face}}$} (plus.east);
\end{tikzpicture}}
\caption{ContextPose. A temporal convolution encodes each window's pose sequence, and a query formed from the projected context and a learned region embedding attends over the window's time steps; the convolution and the cross-attention share their weights across the three windows. A Transformer layer relates the three window summaries, which are then averaged and decoded to valence and arousal. An optional gated face stream adds a zero-initialised correction, and the output is a residual on a context-lookup offset.}
\label{fig:arch}
\end{figure}
%-----------------------------------------------------------------------------
\subsection{Problem formulation and inputs}
\label{sec:inputs}
We organise each clip into three event-centred intervals, indexed by $r\in\regions={\text{pre},\text{event},\text{post}}$: a 5,s pre-event interval, the event interval itself, and a 5,s post-event interval. Clips lasting 10,s or less are omitted because they contain no event frames under this partition. The pose input for each interval is $\mathbf{P}_r\in\R^{T\times J\times F}$, while the overall clip is associated with an event-context vector $\vc\in\R^{d_c}$. If the facial branch is active, each interval additionally provides face embeddings $\mathbf{X}^f_r\in\R^{T\times D}$ and a binary validity mask $\mathbf{m}_r\in{0,1}^{T}$. The network outputs two unconstrained values $\vy\in\R^2$, corresponding to valence and arousal, whose interpretation is specified by the selected learning objective.
% \paragraph{Pose representation.}
% \We retain $J=11$ upper-body COCO landmarks, comprising the nose, eyes, ears, shoulders, elbows, and wrists, extracted at $f_s=25$,fps. Let $\mathbf{p}_{n,j}\in\R^2$ denote the image coordinates of joint $j$ in frame $n$. To reduce sensitivity to camera placement and apparent subject scale, coordinates are centred at the shoulder midpoint $\mathbf{o}_n$ and normalised by the clip-level median shoulder width $w$:
% \begin{equation}
% \tilde{\mathbf{p}}{n,j} = \frac{\mathbf{p}{n,j}-\mathbf{o}_n}{w},
% \end{equation}
% This transformation suppresses differences in image position and apparent camera distance while retaining relative body configuration. To obtain fixed-size inputs from intervals of unequal duration, we resample each interval to $T=32$ temporal steps. Each joint is described by $F=4$ features: two normalised coordinates, detector confidence restricted to $[0,1]$, and movement speed. Missing detections are assigned zero coordinates and confidence. During interpolation, a step uses its two surrounding observations when both exist; if only one exists, that observation is retained, and if neither exists, the step remains zero. This avoids inventing trajectories across missing observations, with the confidence feature explicitly indicating uncertainty.
% Fixed-length resampling alone does not preserve movement rate: an event lasting 20,s and a surrounding interval lasting 5,s would both contain 32 steps despite their different physical durations. We therefore calculate joint speed from the original frame sequence, before temporal resampling, and express it in shoulder widths per second:
% \begin{equation}
% v_{n,j} = \min!\left(f_s,\lVert\tilde{\mathbf{p}}{n,j}-\tilde{\mathbf{p}}{n-1,j}\rVert_2,; v_{\max}\right), \qquad v_{\max}=20.
% \end{equation}
% If a valid previous observation is unavailable, the next observed frame is used; joints lacking a valid neighbouring observation receive zero speed. The limit $v_{\max}$ reduces sensitivity to abrupt tracking errors. For each resampled step, the speed feature is the average over observed frames in a centred span of approximately $N_r/T$ original frames, where $N_r$ is the interval length. Accordingly, the feature retains a consistent real-time interpretation across intervals.
\subsection{Temporal pose encoder}
\label{sec:pose-encoder}
For each region, we concatenate the joint-level features at every step to form $\mX_r\in\R^{T\times JF}$. Two one-dimensional temporal convolutional layers, each with kernel size 3 and zero padding, transform the sequence into $W$-dimensional temporal tokens:
\begin{equation}
\mZ_r = \GELU!\big(\mathrm{Conv}_2\big(\GELU(\mathrm{Conv}1(\mX_r))\big)\big)\in\R^{T\times W},
\qquad
\big[\mathrm{Conv}(\mX)\big]t = \vb + \sum{k=-1}^{1}\mW_k,\mathbf{x}{t+k},
\end{equation}
Boundary inputs satisfy $\mathbf{x}0=\mathbf{x}{T+1}=\mathbf{0}$. The first convolution has weights $\mW_k\in\R^{W\times JF}$, whereas the second uses $\mW_k\in\R^{W\times W}$. Stacking these layers yields a five-step receptive field, enabling each token $\mathbf{z}_{r,t}$ to encode local motion rather than an isolated pose. Parameters are shared across the three regions. No explicit temporal positional embedding is introduced: ordering is represented locally through convolution, while the subsequent within-region attention is otherwise permutation-invariant over the generated tokens.
\subsection{Context encoder}
\label{sec:context}
We investigate three alternative representations of environmental context, each providing a different level of semantic detail. The first consists of four annotator-defined binary indicators describing social interaction, auditory events, activity or object changes, and visual or media events ($d_c=4$); clips without annotated events receive an all-zero vector. The second expresses these indicators as natural-language event descriptions and encodes the sentences using a frozen CLIP ViT-L/14 text encoder, followed by $\ell_2$ normalisation ($d_c=768$). As this text is generated from the event indicators, it does not add independent scene content; only nine unique sentences occur in the development clips. The third representation instead derives descriptions from the video itself. Qwen3-Omni generates accounts of the scene, the child's behaviour, and audible surroundings over successive 5--6,s segments. These descriptions are encoded using all-MiniLM-L6-v2, averaged across segments, and $\ell_2$-normalised to obtain a clip-level vector ($d_c=384$). Each representation is mapped to the common latent width using a linear projection:
\begin{equation}
\vchat = \mW_c,\vc + \vb_c\in\R^{W}.
\end{equation}
In addition, each temporal region is assigned a learned embedding $\ve_r\in\R^W$, with parameters initialised from $\mathcal{N}(0,0.02^2)$. Because the cross-attention module shares parameters across regions, these embeddings provide the region-specific identity needed to distinguish pre-event, event, and post-event queries.
\subsection{Context-queried temporal attention}
\label{sec:cross-attention}
The central fusion mechanism is context-conditioned temporal cross-attention. For each region, the model constructs a query from the projected environmental context and the corresponding region embedding; keys and values come exclusively from that region's pose tokens. The resulting single query vector is defined as:
\begin{equation}
\vq_r = \vchat + \ve_r\in\R^{1\times W}.
\end{equation}
We apply multi-head attention with $H=4$ heads, each of dimension $d_h=W/H$. For head $i$, the query, key, value, and attention weights are computed as follows:
\begin{align}
\mathbf{Q}{r,i} &= \vq_r\mW^{Q}i, \qquad
\mathbf{K}{r,i} = \mZ_r\mW^{K}i, \qquad
\mathbf{V}{r,i} = \mZ_r\mW^{V}i, \
\bm{\alpha}{r,i} &= \softmax!\left(\frac{\mathbf{Q}{r,i}\mathbf{K}_{r,i}^{\top}}{\sqrt{d_h}}\right)\in\R^{1\times T}. \label{eq:attn}
\end{align}
The head outputs are concatenated and linearly projected into the region-level representation:
\begin{equation}
\vs_r = \big[\bm{\alpha}{r,1}\mathbf{V}{r,1};\Vert;\cdots;\Vert;\bm{\alpha}{r,H}\mathbf{V}{r,H}\big],\mW^{O}\in\R^{W},
\end{equation}
Here, $\mW^{Q}_i,\mW^{K}_i,\mW^{V}_i\in\R^{W\times d_h}$ and $\mW^{O}\in\R^{W\times W}$, with bias terms excluded from the notation. The same attention projections are reused across all three regions.
As shown by Equation~\eqref{eq:attn}, environmental context influences the attention distribution rather than being inserted directly into the pose-derived values. For each head, the region summary is a weighted average of projected pose tokens, and the context vector $\vc$ affects which tokens receive the greatest weights. Consequently, an auditory event and a social interaction may lead the model to emphasise different periods of the same behavioural sequence. This mechanism produces a fixed-dimensional summary $\vs_r$ for every region, independent of its original temporal extent.
\subsection{Inter-region Transformer and prediction head}
\label{sec:region-transformer}
The regional summaries are assembled into $\mS=[\vs_{\text{pre}};\vs_{\text{event}};\vs_{\text{post}}]\in\R^{3\times W}$. A single post-normalisation Transformer encoder with $H=4$ attention heads models relationships among these representations:
\begin{align}
\mS' &= \LN!\big(\mS + \Drop(\mathrm{MHSA}(\mS))\big), \
\mS'' &= \LN!\big(\mS' + \Drop(\mathrm{FFN}(\mS'))\big), \qquad
\mathrm{FFN}(\mathbf{x}) = \mW_2,\Drop!\big(\ReLU(\mW_1\mathbf{x}+\vb_1)\big)+\vb_2.
\end{align}
The operation $\mathrm{MHSA}$ denotes self-attention over the three region vectors; the feed-forward sublayer has width $2W$, and attention-weight dropout is applied. Through this operation, the representation of a given region can incorporate information from the others. For example, behaviour during the event can be interpreted in relation to the child's movements immediately before and afterward. We obtain a clip-level representation by averaging the transformed regional outputs:
\begin{equation}
\vh = \frac{1}{3}\sum_{r\in\regions}\mS''_r\in\R^{W}.
\end{equation}
We do not apply positional encodings at this stage. Since self-attention followed by mean aggregation is invariant to permutations of the three input rows, the ordering of the regional vectors is not explicitly available to the final representation. The learned region embeddings introduced at the preceding attention stage are therefore the only source of region identity.
An optional \emph{context skip} connection modifies the fused representation as $\vh\leftarrow\vh+\vchat$, allowing context to influence the output directly as well as through temporal selection. A two-layer MLP, with input dropout and an intermediate dimension of $W/2$, maps the resulting representation to the pose-branch valence and arousal scores:
\begin{equation}
\vy_{\text{pose}} = \mW_4,\GELU!\big(\mW_3,\Drop_{p_h}(\vh)+\vb_3\big)+\vb_4\in\R^{2}.
\end{equation}
\subsection{Gated face stream}
\label{sec:face}
Although facial expression can provide strong evidence of valence, its reliability varies considerably in naturalistic recordings of children. Face crops may be occluded, blurred, poorly illuminated, or affected by head rotation. We therefore model facial information through a separate auxiliary branch that contributes an additive correction to the pose-based estimate only when adequate facial observations are available.
For each of the $T$ temporal positions in a region, we extract the face crop from the nearest video frame. The crop is centred using the tracked facial landmarks, covers a square region of width 0.9 shoulder widths, and is resized to 224,px. A frozen facial-expression ViT pretrained on FER2013 produces a $D=768$-dimensional [CLS] embedding $\mathbf{x}^f_{r,t}\in\R^{D}$. A binary mask $m_{r,t}$ identifies valid crops according to three criteria: (i) the pose tracker detects the nose and both eyes, indicating a frontal face; (ii) YuNet independently detects a face at that location with confidence at least 0.8; and (iii) the crop exceeds a mean grey-level threshold of 40 and a Laplacian-variance threshold of 10. These checks reject unreliable detections, including cases in which the pose estimator reports facial landmarks despite occlusion or a lowered head. Invalid steps have their features set to zero.
We first project each face embedding through layer normalisation, a linear transformation, and GELU activation:
\begin{equation}
\vu_{r,t} = \GELU!\big(\mW_f,\LN(\mathbf{x}^f_{r,t})+\vb_f\big)\in\R^{32},
\end{equation}
The resulting step features are aggregated using a mask-weighted mean within each region:
\begin{equation}
\bar{\vu}r = \frac{\sum{t=1}^{T} m_{r,t},\vu_{r,t}}{\max!\big(1,\sum_{t=1}^{T} m_{r,t}\big)},
\end{equation}
The denominator $\max(1,\cdot)$ ensures that a region with no valid facial frames has a zero summary. The three 32-dimensional summaries are concatenated and passed to a two-layer prediction head $\psi$ ($96\to W/2\to k$), generating the following correction:
\begin{equation}
\bm{\delta} = \psi\big(\Drop_{p_h}([\bar{\vu}{\text{pre}};\bar{\vu}{\text{event}};\bar{\vu}{\text{post}}])\big)\in\R^{k},
\qquad
\bm{\Delta}{\text{face}} = \mathbb{I}\Big[\textstyle\sum_{r,t} m_{r,t}>0\Big]\cdot
\begin{cases}
[\delta_1,,0]^{\top}, & k=1 \text{ (valence only)},\
\bm{\delta}, & k=2 \text{ (both targets)}.
\end{cases}
\label{eq:face}
\end{equation}
We introduce three safeguards to limit the effect of low-quality facial information. First, the final layer of $\psi$ is initialised to zero, so the model initially reproduces the pose-and-context prediction and only later learns an additional facial contribution. Second, the indicator in Equation~\eqref{eq:face} forces $\bm{\Delta}{\text{face}}=\mathbf{0}$ when no valid facial step exists anywhere in the clip, preventing a spurious correction from the head's bias. Third, during training, the complete face branch is randomly removed with probability $p_f$ by applying $m{r,t}\leftarrow m_{r,t}(1-\xi)$ for all regions and steps, where $\xi\sim\mathrm{Bernoulli}(p_f)$. This face dropout also makes the correction zero and encourages the remaining modalities to support the prediction independently.
\subsection{Residual prediction on a context lookup}
\label{sec:residual}
Because event-indicator combinations are informative about the observed annotation distribution, a simple context-conditioned lookup provides a useful baseline. Given the small development sample of approximately 150 clips, we train ContextPose to estimate deviations from this baseline rather than requiring its neural branches to learn the full prediction directly:
\begin{equation}
\vy = \vy_{\text{pose}} + \bm{\Delta}{\text{face}} + \vb{g(\vc)}.
\label{eq:output}
\end{equation}
In Equation~\eqref{eq:output}, $g(\vc)$ identifies the event-indicator combination associated with the clip, and $\vb_{g}\in\R^2$ denotes its fixed baseline offset. Offsets are estimated exclusively from training examples within each cross-validation fold and are not updated by gradient descent, thereby excluding evaluation data from baseline estimation. For vote-based learning, each offset equals the log-odds of the group's mean vote proportion $\bar{\pi}_{g}$. We clip this proportion to $[\epsilon,1-\epsilon]$, where $\epsilon=0.02$, to avoid infinite log-odds:
\begin{equation}
\vb_{g} = \log\frac{\bar{\pi}{g}}{1-\bar{\pi}{g}}.
\end{equation}
For regression, the offset is instead the median target value for the corresponding group. If fewer than three training clips are available for a group, the global training-set estimate is used. Disabling the lookup sets the offset to zero; the prediction head's bias then supplies any learned constant component, and no label statistics are injected through this branch.
\subsection{Output parameterisation and objective}
\label{sec:objective}
Three annotators assessed each clip, assigning positive, neutral, or negative valence and either high or low arousal. Given the moderate inter-annotator agreement, our primary objective retains the distribution of votes instead of collapsing them to a single consensus class. For each dimension, $\pi\in[0,1]$ denotes the proportion of known votes for positive valence or high arousal. The prediction $\hat{y}$ is interpreted as the logit of that vote probability, and training minimises binary cross-entropy with the soft vote proportion for both outputs, averaged over the batch:
\begin{equation}
\mathcal{L} = -\sum_{k\in{\text{val},,\text{aro}}}\Big[\pi_k\log\sigma(\hat{y}_k)+(1-\pi_k)\log\big(1-\sigma(\hat{y}_k)\big)\Big].
\end{equation}
During inference, $\sigma(\hat{y}_k)$ gives the estimated vote probability, used to distinguish positive from non-positive valence or high from low arousal after applying a threshold. Thresholds are chosen to maximise balanced accuracy using only model-fitting and selection data, never evaluation clips. For the alternative regression objective, the two outputs represent consensus valence and arousal directly and are optimised using mean squared error.
\subsection{Stream ablations}
\label{sec:ablations}
To assess the contribution of each modality, we evaluate stream-removal variants that retain the same basic prediction framework while disabling selected inputs:
\begin{align}
\text{pose only:}\quad & \vy = \vy_{\text{pose}}\big|{\vchat=\mathbf{0}} + \bm{\Delta}{\text{face}} + \vb, \
\text{context only:}\quad & \vy = \mathrm{MLP}\big(\Drop_{p_h}(\vchat)\big) + \bm{\Delta}{\text{face}} + \vb, \
\text{face only:}\quad & \vy = \bm{\Delta}{\text{face}} + \vb.
\end{align}
In the pose-only variant, the projected context is set to zero, leaving the learned region embeddings to provide the cross-attention queries. In the context-only variant, the pose pathway is bypassed and the prediction MLP operates directly on the projected context. The face-only configuration removes the pose prediction and retains the facial correction where that branch is enabled. Across these experiments, $\vb$ denotes either the fixed context-dependent offset $\vb_{g(\vc)}$ or zero when the lookup is inactive. Thus, when the lookup is enabled, the nominal face-only condition still incorporates event information through its baseline offset.
\subsection{Implementation details}
\label{sec:impl}
We keep the network compact to limit overfitting on the development set of approximately 150 clips. Table~\ref{tab:hparams} summarises the architectural and optimisation settings. With latent width $W=64$, the indicator-context configuration contains 73.6k trainable parameters; using VLM or CLIP features increases this to 98.0k and 122.5k, respectively, primarily due to the input projection. The optional face branch introduces another 29.3k parameters, largely from mapping the 768-dimensional face embedding to 32 dimensions.
We optimise all variants using AdamW with learning rate $3\times10^{-4}$, weight decay 0.05, and batch size 16. Training lasts at most 60 epochs for indicator and CLIP contexts or 150 epochs for VLM context, with early stopping after 10 epochs without validation improvement. The selected checkpoint minimises validation cross-entropy in the vote-prediction setting and validation mean absolute error in regression. To augment each training sequence, we apply random rotations within $\pm15^\circ$, scale factors in $[0.9,1.1]$, and Gaussian coordinate noise of standard deviation $\sigma=0.02$ shoulder widths. Horizontal reflection is applied with probability 0.5, reversing the horizontal coordinate and exchanging left and right joints. All regions of a clip receive the same transform, and missing joints remain unchanged. For final evaluation, we ensemble five independently seeded models, averaging their prediction scores and using a majority vote of their binary decisions.
\begin{table}[h]
\centering
\caption{Hyper-parameters of ContextPose.}
\label{tab:hparams}
\small
\begin{tabular}{@{}llr@{}}
\toprule
Component & Hyper-parameter & Value \
\midrule
Input & Steps per window $T$ / joints $J$ / features $F$ & 32 / 11 / 4 \
& Context dimension $d_c$ (indicators / CLIP / VLM) & 4 / 768 / 384 \
Pose encoder & Conv layers / kernel / activation & 2 / 3 / GELU \
Model width & $W$ & 64 \
Cross-attention & Heads $H$ / head width $d_h$ & 4 / 16 \
Region Transformer & Layers / heads / FFN width & 1 / 4 / $2W$ \
Face stream & Input $D$ / encoding width & 768 / 32 \
Head & Hidden width & $W/2$ \
Regularisation & Transformer dropout / head dropout $p_h$ / face dropout $p_f$ & 0.3 / 0.3 / 0.3 \
\midrule
Optimisation & Optimiser / learning rate / weight decay & AdamW / $3{\times}10^{-4}$ / 0.05 \
& Batch size / max.\ epochs / early-stopping patience & 16 / 60 / 10 \
& Ensemble size (seeds) & 5 \
\midrule
Parameters & Without / with face stream (indicator context) & 73.6k / 102.9k \
\bottomrule
\end{tabular}
\end{table}
\end{document}
This is what I'm currently working with. For parts where we describe joint features as an input stream, joints features can also be replace by subject aware encoder directly from input videos such as EmotionClip
</pasted_content id="3ab9">

. So its either pose or EmotionClip features

---


**Claude**

I'm collecting the details of how EmotionCLIP features were extracted in earlier sessions, so the methodology describes them accurately. After that I'll edit the document.

---


**Claude**

I've updated [contextpose_methodology.tex](reports/contextpose_methodology.tex) so the input stream is either pose features or EmotionCLIP frame embeddings. Everything after the input is shared. I built it on the version you pasted, since that's newer than what was on disk. The earlier draft is backed up at `/private/tmp/claude-501/-Users-henryansah--claude-spares-spare-5WONKs/67cd9486-2e25-4f58-9bb8-bc406f97875a/scratchpad/contextpose_methodology.before_eclip.tex`. There's no LaTeX compiler installed, so I checked braces, environments and references with a script but haven't compiled it or looked at the figure.

**What changed**
- **Wording:** "pose features/tokens" is now "behaviour features/tokens" wherever the text describes the input stream. The head output $\vy_{\text{pose}}$ is now $\vy_{\text{beh}}$, and the "pose only" ablation is now "behaviour only". The model is still called ContextPose.
- **Problem formulation:** the input is now a single $\mX_r\in\R^{T\times D_x}$, followed by two new paragraphs:
  - **Pose input:** a short version with 11 joints × 4 features, so $D_x=44$. I left your commented-out pose block exactly as it was.
  - **Subject-aware video input:** EmotionCLIP's ViT-B/32 with its subject mask. This uses the extraction from your earlier experiments: the crop is centred on the child with side 0.875 × the short side, resized to 224 px. The mask comes from the child's joint box padded by 0.3 shoulder widths and extended downward, and it is left empty when the child isn't tracked. The output is a 512-d L2-normalised embedding per step.
- **Encoder:** only the first convolution depends on the input ($\mW_k\in\R^{W\times D_x}$). The section is renamed "Temporal behaviour encoder".
- **Figure:** the input boxes now read "Pose or EmotionCLIP, $\mX\in\R^{T\times D_x}$". I widened them and moved the context arrows to fit, which is why the figure most needs a visual check.
- **Implementation:** pose augmentations apply only to pose input, and EmotionCLIP features are used without augmentation.
- **Parameter counts:** I counted these by building the model.
  - EmotionCLIP input: 163.7k parameters.
  - EmotionCLIP input plus the face stream: 193.0k.
  - Both counts are in the table, along with a $D_x$ row (44 / 513).

**Things to check**
- **The 513th channel is my choice.** I appended the "child tracked" flag to the 512-d embedding, playing the same role as the pose confidence feature. If you'd rather use the plain 512-d embedding, the count becomes 163.5k.
- **This exact setup hasn't been run.** Your EmotionCLIP network experiment (E3) fed the embeddings through the face slot, not the pose slot. If you report results under this methodology, they need a new run. The current code can probably take it as `num_joints=1, joint_features=513`, but pose augmentation would have to be turned off.
- **"0.3 shoulder widths"** is my reading of the extractor's "0.3 × scale". The pose files define that scale as the median shoulder width.
- **The citation is plain text,** "EmotionCLIP (Zhang et al., 2023)", because the document has no bibliography yet. Swap in a `\cite` when you add one.

---


**User**

<pasted_content id="3ab9">
\documentclass[11pt]{article}

\usepackage[margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc}
\usepackage{caption}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}

\captionsetup{font=small,labelfont=bf}

\tikzset{
  box/.style={draw,rounded corners=2pt,align=center,font=\footnotesize,minimum height=7mm,inner sep=3pt,fill=white},
  inp/.style={box,fill=blue!6},
  op/.style={box,fill=gray!8},
  outputnode/.style={box,fill=green!8},
  arr/.style={-{Stealth[length=2mm]},thick},
}

% Notation
\newcommand{\R}{\mathbb{R}}
\newcommand{\regions}{\mathcal{R}}
\newcommand{\vc}{\mathbf{c}}
\newcommand{\vchat}{\hat{\mathbf{c}}}
\newcommand{\ve}{\mathbf{e}}
\newcommand{\vq}{\mathbf{q}}
\newcommand{\vs}{\mathbf{s}}
\newcommand{\vh}{\mathbf{h}}
\newcommand{\vy}{\hat{\mathbf{y}}}
\newcommand{\vb}{\mathbf{b}}
\newcommand{\vu}{\mathbf{u}}
\newcommand{\mZ}{\mathbf{Z}}
\newcommand{\mX}{\mathbf{X}}
\newcommand{\mS}{\mathbf{S}}
\newcommand{\mW}{\mathbf{W}}
\newcommand{\GELU}{\operatorname{GELU}}
\newcommand{\ReLU}{\operatorname{ReLU}}
\newcommand{\LN}{\operatorname{LN}}
\newcommand{\Drop}{\operatorname{Drop}}
\newcommand{\softmax}{\operatorname{softmax}}

\title{ContextPose}
\author{}
\date{}

\begin{document}
\maketitle

\section{Methodology}
\label{sec:method}

ContextPose estimates the valence and arousal reflected in human annotations of short video clips depicting a child before, during, and after an environmental event. The proposed architecture is motivated by the hypothesis that contextual information can help identify the behavioural moments most relevant to an affective response. Rather than merely concatenating context with an aggregated behaviour feature, we use the context representation as an attention query over the temporal behaviour features. In this formulation, context determines how the behaviour sequence is summarised, while the attended values themselves are derived from the child's observed behaviour.

The framework is illustrated in Figure~\ref{fig:arch}. The child's behaviour is described by one of two interchangeable inputs: body-pose features computed from tracked joints, or subject-aware frame embeddings computed directly from the video by EmotionCLIP. Whichever input is used, the sequences from the pre-event, event, and post-event intervals are first encoded independently using a shared temporal convolutional network. Event context, represented either through annotations or a language-based description, is mapped into the same latent space. For each interval, the context embedding is combined with a learned region embedding to query the corresponding temporal behaviour tokens using multi-head cross-attention. The resulting three summaries are integrated by a single-layer Transformer and passed to a prediction head for valence and arousal. When reliable facial observations are available, an auxiliary face branch provides an additive correction. 

%-----------------------------------------------------------------------------
\begin{figure}[t]
\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[every node/.style={text width=1.75cm}]
  % Behaviour columns (pose or EmotionCLIP input)
  \foreach \name/\x/\lbl in {pre/0/pre-event, ev/3/event, post/6/post-event}{
    \node[inp,text width=2.2cm] (P\name) at (\x,0.15) {Pose or EmotionCLIP\\[-1pt]{\scriptsize $\mX_{\text{\lbl}}\in\R^{T\times D_x}$}};
    \node[op]  (T\name) at (\x,-1.4) {Temporal conv.\\[-1pt]{\scriptsize $\to \mZ_r\in\R^{T\times W}$}};
    \node[op,minimum height=11mm] (A\name) at (\x,-2.9) {Cross-attn.\\[-1pt]{\scriptsize $\vq_r=\vchat+\ve_r$}};
    \draw[arr] (P\name) -- (T\name);
    \draw[arr] (T\name) -- node[right,font=\scriptsize]{$\mZ_r$} (A\name);
  }

  % Context path
  \node[inp,text width=2.3cm] (C) at (-2.9,1.35) {Context $\vc$\\[-1pt]{\scriptsize indicators / CLIP / VLM}};
  \node[op] (Wc) at (-0.15,1.35) {Linear\\[-1pt]{\scriptsize $\vchat=\mW_c\vc+\vb_c$}};
  \draw[arr] (C) -- (Wc);
  \draw[thick] (Wc.east) -- (7.4,1.35);
  \foreach \name/\x in {pre/1.4, ev/4.4, post/7.4}{
    \draw[arr] (\x,1.35) |- (A\name.east);
  }

  % Region transformer and head
  \node[op,text width=7.6cm] (RT) at (3,-4.6) {Inter-region Transformer over $[\vs_{\text{pre}};\vs_{\text{event}};\vs_{\text{post}}]$, then mean over regions $\to \vh$};
  \foreach \name in {pre,ev,post}{ \draw[arr] (A\name) -- node[right,font=\scriptsize]{$\vs_r$} (A\name |- RT.north); }
  \node[op,text width=3.2cm] (H) at (3,-5.85) {MLP head $W\!\to\!W/2\!\to\!2$};
  \draw[arr] (RT) -- (H);
  \node[draw,circle,inner sep=1pt,text width=0.4cm,font=\small] (plus) at (3,-7.0) {$+$};
  \draw[arr] (H) -- node[right,font=\scriptsize]{$\vy_{\text{beh}}$} (plus);
  \node[inp,text width=2.6cm] (bg) at (-0.2,-7.0) {Offset $\vb_{g(\vc)}$\\[-1pt]{\scriptsize context lookup}};
  \draw[arr] (bg) -- (plus);
  \node[outputnode,text width=2.6cm] (y) at (3,-8.1) {$\vy=[\hat y_{\text{val}},\hat y_{\text{aro}}]$};
  \draw[arr] (plus) -- (y);

  % Face stream
  \node[inp,text width=2.4cm] (F) at (9.7,0) {Face features $\mathbf{x}^{f}_{r,t}$, gate $m_{r,t}$};
  \node[op,text width=2.4cm] (FE) at (9.7,-1.4) {LN $\to$ Linear $\to$ GELU\\[-1pt]{\scriptsize $\vu_{r,t}\in\R^{32}$}};
  \node[op,text width=2.4cm,minimum height=11mm] (FP) at (9.7,-2.9) {Gated mean per region\\[-1pt]{\scriptsize $\bar{\vu}_r$}};
  \node[op,text width=2.4cm] (FH) at (9.7,-4.6) {Face head $\psi$\\[-1pt]{\scriptsize zero-initialised}};
  \draw[arr] (F) -- (FE);
  \draw[arr] (FE) -- (FP);
  \draw[arr] (FP) -- (FH);
  \draw[arr] (FH.south) |- node[pos=0.75,above,font=\scriptsize]{$\bm{\Delta}_{\text{face}}$} (plus.east);
\end{tikzpicture}}
\caption{ContextPose. Each window's behaviour sequence, given either as pose features or as subject-aware EmotionCLIP frame embeddings, is encoded by a temporal convolution, and a query formed from the projected context and a learned region embedding attends over the window's time steps; the convolution and the cross-attention share their weights across the three windows. A Transformer layer relates the three window summaries, which are then averaged and decoded to valence and arousal. An optional gated face stream adds a zero-initialised correction, and the output is a residual on a context-lookup offset.}
\label{fig:arch}
\end{figure}
%-----------------------------------------------------------------------------

\subsection{Problem formulation and inputs}
\label{sec:inputs}

We organise each clip into three event-centred intervals, indexed by $r\in\regions=\{\text{pre},\text{event},\text{post}\}$: a 5\,s pre-event interval, the event interval itself, and a 5\,s post-event interval. Clips lasting 10\,s or less are omitted because they contain no event frames under this partition. Each interval is resampled to $T=32$ temporal steps, and its behaviour input is a sequence of step vectors $\mX_r\in\R^{T\times D_x}$, built either from pose or from EmotionCLIP as described below. The overall clip is associated with an event-context vector $\vc\in\R^{d_c}$. If the facial branch is active, each interval additionally provides face embeddings $\mathbf{X}^f_r\in\R^{T\times D}$ and a binary validity mask $\mathbf{m}_r\in\{0,1\}^{T}$. The network outputs two unconstrained values $\vy\in\R^2$, corresponding to valence and arousal, whose interpretation is specified by the selected learning objective.

% \paragraph{Pose representation.}
 % \We retain $J=11$ upper-body COCO landmarks, comprising the nose, eyes, ears, shoulders, elbows, and wrists, extracted at $f_s=25$\,fps. Let $\mathbf{p}_{n,j}\in\R^2$ denote the image coordinates of joint $j$ in frame $n$. To reduce sensitivity to camera placement and apparent subject scale, coordinates are centred at the shoulder midpoint $\mathbf{o}_n$ and normalised by the clip-level median shoulder width $w$:

% \begin{equation}
%   \tilde{\mathbf{p}}_{n,j} = \frac{\mathbf{p}_{n,j}-\mathbf{o}_n}{w},
% \end{equation}

% This transformation suppresses differences in image position and apparent camera distance while retaining relative body configuration. To obtain fixed-size inputs from intervals of unequal duration, we resample each interval to $T=32$ temporal steps. Each joint is described by $F=4$ features: two normalised coordinates, detector confidence restricted to $[0,1]$, and movement speed. Missing detections are assigned zero coordinates and confidence. During interpolation, a step uses its two surrounding observations when both exist; if only one exists, that observation is retained, and if neither exists, the step remains zero. This avoids inventing trajectories across missing observations, with the confidence feature explicitly indicating uncertainty.

% Fixed-length resampling alone does not preserve movement rate: an event lasting 20\,s and a surrounding interval lasting 5\,s would both contain 32 steps despite their different physical durations. We therefore calculate joint speed from the original frame sequence, before temporal resampling, and express it in shoulder widths per second:

% \begin{equation}
%   v_{n,j} = \min\!\left(f_s\,\lVert\tilde{\mathbf{p}}_{n,j}-\tilde{\mathbf{p}}_{n-1,j}\rVert_2,\; v_{\max}\right), \qquad v_{\max}=20.
% \end{equation}

% If a valid previous observation is unavailable, the next observed frame is used; joints lacking a valid neighbouring observation receive zero speed. The limit $v_{\max}$ reduces sensitivity to abrupt tracking errors. For each resampled step, the speed feature is the average over observed frames in a centred span of approximately $N_r/T$ original frames, where $N_r$ is the interval length. Accordingly, the feature retains a consistent real-time interpretation across intervals.

\paragraph{Pose input.} 


\paragraph{Subject-aware video input.} Alternatively, each step is described directly from the video by EmotionCLIP (Zhang et al., 2023), a CLIP ViT-B/32 image encoder trained on vlog videos and their transcripts to represent the emotional state of a designated person. EmotionCLIP is subject-aware: alongside the RGB frame, it receives a binary mask marking the person to be read, which is average-pooled over each $32\times32$ patch and added to that patch's positional embedding. The encoder therefore attends to the subject's body and face while still seeing the surrounding scene. At each of the $T$ steps, we take the corresponding video frame, crop a square centred on the child with side 0.875 times the frame's shorter side, and resize it to 224\,px with ImageNet normalisation. EmotionCLIP was trained with person boxes from an object detector; we instead build the mask from the child's tracked joints, as their bounding box padded by 0.3 shoulder widths and extended downwards, because the 11 joints cover mainly the upper body. Deriving the mask from the child's own track keeps it on the child rather than on an adult or a person shown on a screen. When the child is not tracked within two frames of the step, the mask is left empty rather than guessed. The frozen encoder's output is $\ell_2$-normalised to a 512-dimensional frame embedding, and a binary flag recording whether the child was tracked is appended, giving $D_x=513$. Like the confidence feature of the pose input, this flag tells the encoder which steps are less reliable. We use EmotionCLIP's per-frame encoder rather than its own temporal Transformer, so that our temporal encoder and context-queried attention operate on the same $T$ steps for both inputs. With this input, pose only determines where the crop and mask are placed; joint coordinates do not enter the features.

\subsection{Temporal behaviour encoder}
\label{sec:pose-encoder}

For each region, the behaviour input $\mX_r\in\R^{T\times D_x}$ holds one vector per step: the concatenated joint features for pose, or the flagged frame embedding for EmotionCLIP. Two one-dimensional temporal convolutional layers, each with kernel size 3 and zero padding, transform the sequence into $W$-dimensional temporal tokens:

\begin{equation}
  \mZ_r = \GELU\!\big(\mathrm{Conv}_2\big(\GELU(\mathrm{Conv}_1(\mX_r))\big)\big)\in\R^{T\times W},
  \qquad
  \big[\mathrm{Conv}(\mX)\big]_t = \vb + \sum_{k=-1}^{1}\mW_k\,\mathbf{x}_{t+k},
\end{equation}

Boundary inputs satisfy $\mathbf{x}_0=\mathbf{x}_{T+1}=\mathbf{0}$. The first convolution has weights $\mW_k\in\R^{W\times D_x}$, whereas the second uses $\mW_k\in\R^{W\times W}$. Only the first convolution depends on the choice of input; every later layer is identical for pose and EmotionCLIP. Stacking these layers yields a five-step receptive field, enabling each token $\mathbf{z}_{r,t}$ to encode local change across neighbouring steps rather than a single isolated observation. Parameters are shared across the three regions. No explicit temporal positional embedding is introduced: ordering is represented locally through convolution, while the subsequent within-region attention is otherwise permutation-invariant over the generated tokens.

\subsection{Context encoder}
\label{sec:context}

We investigate three alternative representations of environmental context, each providing a different level of semantic detail. The first consists of four annotator-defined binary indicators describing social interaction, auditory events, activity or object changes, and visual or media events ($d_c=4$); clips without annotated events receive an all-zero vector. The second expresses these indicators as natural-language event descriptions and encodes the sentences using a frozen CLIP ViT-L/14 text encoder, followed by $\ell_2$ normalisation ($d_c=768$). As this text is generated from the event indicators, it does not add independent scene content; only nine unique sentences occur in the development clips. The third representation instead derives descriptions from the video itself. Qwen3-Omni generates accounts of the scene, the child's behaviour, and audible surroundings over successive 5--6\,s segments. These descriptions are encoded using all-MiniLM-L6-v2, averaged across segments, and $\ell_2$-normalised to obtain a clip-level vector ($d_c=384$). Each representation is mapped to the common latent width using a linear projection:

\begin{equation}
  \vchat = \mW_c\,\vc + \vb_c\in\R^{W}.
\end{equation}

In addition, each temporal region is assigned a learned embedding $\ve_r\in\R^W$, with parameters initialised from $\mathcal{N}(0,0.02^2)$. Because the cross-attention module shares parameters across regions, these embeddings provide the region-specific identity needed to distinguish pre-event, event, and post-event queries.

\subsection{Context-queried temporal attention}
\label{sec:cross-attention}

The central fusion mechanism is context-conditioned temporal cross-attention. For each region, the model constructs a query from the projected environmental context and the corresponding region embedding; keys and values come exclusively from that region's behaviour tokens. The resulting single query vector is defined as:

\begin{equation}
  \vq_r = \vchat + \ve_r\in\R^{1\times W}.
\end{equation}

We apply multi-head attention with $H=4$ heads, each of dimension $d_h=W/H$. For head $i$, the query, key, value, and attention weights are computed as follows:

\begin{align}
  \mathbf{Q}_{r,i} &= \vq_r\mW^{Q}_i, \qquad
  \mathbf{K}_{r,i} = \mZ_r\mW^{K}_i, \qquad
  \mathbf{V}_{r,i} = \mZ_r\mW^{V}_i, \\
  \bm{\alpha}_{r,i} &= \softmax\!\left(\frac{\mathbf{Q}_{r,i}\mathbf{K}_{r,i}^{\top}}{\sqrt{d_h}}\right)\in\R^{1\times T}. \label{eq:attn}
\end{align}

The head outputs are concatenated and linearly projected into the region-level representation:

\begin{equation}
  \vs_r = \big[\bm{\alpha}_{r,1}\mathbf{V}_{r,1}\;\Vert\;\cdots\;\Vert\;\bm{\alpha}_{r,H}\mathbf{V}_{r,H}\big]\,\mW^{O}\in\R^{W},
\end{equation}

Here, $\mW^{Q}_i,\mW^{K}_i,\mW^{V}_i\in\R^{W\times d_h}$ and $\mW^{O}\in\R^{W\times W}$, with bias terms excluded from the notation. The same attention projections are reused across all three regions.

As shown by Equation~\eqref{eq:attn}, environmental context influences the attention distribution rather than being inserted directly into the behaviour-derived values. For each head, the region summary is a weighted average of projected behaviour tokens, and the context vector $\vc$ affects which tokens receive the greatest weights. Consequently, an auditory event and a social interaction may lead the model to emphasise different periods of the same behavioural sequence. This mechanism produces a fixed-dimensional summary $\vs_r$ for every region, independent of its original temporal extent.

\subsection{Inter-region Transformer and prediction head}
\label{sec:region-transformer}

The regional summaries are assembled into $\mS=[\vs_{\text{pre}};\vs_{\text{event}};\vs_{\text{post}}]\in\R^{3\times W}$. A single post-normalisation Transformer encoder with $H=4$ attention heads models relationships among these representations:

\begin{align}
  \mS' &= \LN\!\big(\mS + \Drop(\mathrm{MHSA}(\mS))\big), \\
  \mS'' &= \LN\!\big(\mS' + \Drop(\mathrm{FFN}(\mS'))\big), \qquad
  \mathrm{FFN}(\mathbf{x}) = \mW_2\,\Drop\!\big(\ReLU(\mW_1\mathbf{x}+\vb_1)\big)+\vb_2.
\end{align}

The operation $\mathrm{MHSA}$ denotes self-attention over the three region vectors; the feed-forward sublayer has width $2W$, and attention-weight dropout is applied. Through this operation, the representation of a given region can incorporate information from the others. For example, behaviour during the event can be interpreted in relation to the child's behaviour immediately before and afterward. We obtain a clip-level representation by averaging the transformed regional outputs:

\begin{equation}
  \vh = \frac{1}{3}\sum_{r\in\regions}\mS''_r\in\R^{W}.
\end{equation}

We do not apply positional encodings at this stage. Since self-attention followed by mean aggregation is invariant to permutations of the three input rows, the ordering of the regional vectors is not explicitly available to the final representation. The learned region embeddings introduced at the preceding attention stage are therefore the only source of region identity.

An optional \emph{context skip} connection modifies the fused representation as $\vh\leftarrow\vh+\vchat$, allowing context to influence the output directly as well as through temporal selection. A two-layer MLP, with input dropout and an intermediate dimension of $W/2$, maps the resulting representation to the behaviour-branch valence and arousal scores:

\begin{equation}
  \vy_{\text{beh}} = \mW_4\,\GELU\!\big(\mW_3\,\Drop_{p_h}(\vh)+\vb_3\big)+\vb_4\in\R^{2}.
\end{equation}

\subsection{Gated face stream}
\label{sec:face}

Although facial expression can provide strong evidence of valence, its reliability varies considerably in naturalistic recordings of children. Face crops may be occluded, blurred, poorly illuminated, or affected by head rotation. We therefore model facial information through a separate auxiliary branch that contributes an additive correction to the behaviour-based estimate only when adequate facial observations are available.

For each of the $T$ temporal positions in a region, we extract the face crop from the nearest video frame. The crop is centred using the tracked facial landmarks, covers a square region of width 0.9 shoulder widths, and is resized to 224\,px. A frozen facial-expression ViT pretrained on FER2013 produces a $D=768$-dimensional [CLS] embedding $\mathbf{x}^f_{r,t}\in\R^{D}$. A binary mask $m_{r,t}$ identifies valid crops according to three criteria: (i) the pose tracker detects the nose and both eyes, indicating a frontal face; (ii) YuNet independently detects a face at that location with confidence at least 0.8; and (iii) the crop exceeds a mean grey-level threshold of 40 and a Laplacian-variance threshold of 10. These checks reject unreliable detections, including cases in which the pose estimator reports facial landmarks despite occlusion or a lowered head. Invalid steps have their features set to zero.

We first project each face embedding through layer normalisation, a linear transformation, and GELU activation:

\begin{equation}
  \vu_{r,t} = \GELU\!\big(\mW_f\,\LN(\mathbf{x}^f_{r,t})+\vb_f\big)\in\R^{32},
\end{equation}

The resulting step features are aggregated using a mask-weighted mean within each region:

\begin{equation}
  \bar{\vu}_r = \frac{\sum_{t=1}^{T} m_{r,t}\,\vu_{r,t}}{\max\!\big(1,\sum_{t=1}^{T} m_{r,t}\big)},
\end{equation}

The denominator $\max(1,\cdot)$ ensures that a region with no valid facial frames has a zero summary. The three 32-dimensional summaries are concatenated and passed to a two-layer prediction head $\psi$ ($96\to W/2\to k$), generating the following correction:

\begin{equation}
  \bm{\delta} = \psi\big(\Drop_{p_h}([\bar{\vu}_{\text{pre}};\bar{\vu}_{\text{event}};\bar{\vu}_{\text{post}}])\big)\in\R^{k},
  \qquad
  \bm{\Delta}_{\text{face}} = \mathbb{I}\Big[\textstyle\sum_{r,t} m_{r,t}>0\Big]\cdot
  \begin{cases}
    [\delta_1,\,0]^{\top}, & k=1 \text{ (valence only)},\\
    \bm{\delta}, & k=2 \text{ (both targets)}.
  \end{cases}
  \label{eq:face}
\end{equation}

We introduce three safeguards to limit the effect of low-quality facial information. First, the final layer of $\psi$ is initialised to zero, so the model initially reproduces the behaviour-and-context prediction and only later learns an additional facial contribution. Second, the indicator in Equation~\eqref{eq:face} forces $\bm{\Delta}_{\text{face}}=\mathbf{0}$ when no valid facial step exists anywhere in the clip, preventing a spurious correction from the head's bias. Third, during training, the complete face branch is randomly removed with probability $p_f$ by applying $m_{r,t}\leftarrow m_{r,t}(1-\xi)$ for all regions and steps, where $\xi\sim\mathrm{Bernoulli}(p_f)$. This face dropout also makes the correction zero and encourages the remaining modalities to support the prediction independently.

\subsection{Residual prediction on a context lookup}
\label{sec:residual}

Because event-indicator combinations are informative about the observed annotation distribution, a simple context-conditioned lookup provides a useful baseline. Given the small development sample of approximately 150 clips, we train ContextPose to estimate deviations from this baseline rather than requiring its neural branches to learn the full prediction directly:

\begin{equation}
  \vy = \vy_{\text{beh}} + \bm{\Delta}_{\text{face}} + \vb_{g(\vc)}.
  \label{eq:output}
\end{equation}

In Equation~\eqref{eq:output}, $g(\vc)$ identifies the event-indicator combination associated with the clip, and $\vb_{g}\in\R^2$ denotes its fixed baseline offset. Offsets are estimated exclusively from training examples within each cross-validation fold and are not updated by gradient descent, thereby excluding evaluation data from baseline estimation. For vote-based learning, each offset equals the log-odds of the group's mean vote proportion $\bar{\pi}_{g}$. We clip this proportion to $[\epsilon,1-\epsilon]$, where $\epsilon=0.02$, to avoid infinite log-odds:

\begin{equation}
  \vb_{g} = \log\frac{\bar{\pi}_{g}}{1-\bar{\pi}_{g}}.
\end{equation}

For regression, the offset is instead the median target value for the corresponding group. If fewer than three training clips are available for a group, the global training-set estimate is used. Disabling the lookup sets the offset to zero; the prediction head's bias then supplies any learned constant component, and no label statistics are injected through this branch.

\subsection{Output parameterisation and objective}
\label{sec:objective}

Three annotators assessed each clip, assigning positive, neutral, or negative valence and either high or low arousal. Given the moderate inter-annotator agreement, our primary objective retains the distribution of votes instead of collapsing them to a single consensus class. For each dimension, $\pi\in[0,1]$ denotes the proportion of known votes for positive valence or high arousal. The prediction $\hat{y}$ is interpreted as the logit of that vote probability, and training minimises binary cross-entropy with the soft vote proportion for both outputs, averaged over the batch:

\begin{equation}
  \mathcal{L} = -\sum_{k\in\{\text{val},\,\text{aro}\}}\Big[\pi_k\log\sigma(\hat{y}_k)+(1-\pi_k)\log\big(1-\sigma(\hat{y}_k)\big)\Big].
\end{equation}

During inference, $\sigma(\hat{y}_k)$ gives the estimated vote probability, used to distinguish positive from non-positive valence or high from low arousal after applying a threshold. Thresholds are chosen to maximise balanced accuracy using only model-fitting and selection data, never evaluation clips. For the alternative regression objective, the two outputs represent consensus valence and arousal directly and are optimised using mean squared error.

\subsection{Stream ablations}
\label{sec:ablations}

To assess the contribution of each modality, we evaluate stream-removal variants that retain the same basic prediction framework while disabling selected inputs:

\begin{align}
  \text{behaviour only:}\quad & \vy = \vy_{\text{beh}}\big|_{\vchat=\mathbf{0}} + \bm{\Delta}_{\text{face}} + \vb, \\
  \text{context only:}\quad & \vy = \mathrm{MLP}\big(\Drop_{p_h}(\vchat)\big) + \bm{\Delta}_{\text{face}} + \vb, \\
  \text{face only:}\quad & \vy = \bm{\Delta}_{\text{face}} + \vb.
\end{align}

In the behaviour-only variant, the projected context is set to zero, leaving the learned region embeddings to provide the cross-attention queries; it can be run with either pose or EmotionCLIP input. In the context-only variant, the behaviour pathway is bypassed and the prediction MLP operates directly on the projected context. The face-only configuration removes the behaviour prediction and retains the facial correction where that branch is enabled. Across these experiments, $\vb$ denotes either the fixed context-dependent offset $\vb_{g(\vc)}$ or zero when the lookup is inactive. Thus, when the lookup is enabled, the nominal face-only condition still incorporates event information through its baseline offset.

\subsection{Implementation details}
\label{sec:impl}

We keep the network compact to limit overfitting on the development set of approximately 150 clips. Table~\ref{tab:hparams} summarises the architectural and optimisation settings. With latent width $W=64$, pose input, and the indicator context, the model contains 73.6k trainable parameters; using VLM or CLIP features increases this to 98.0k and 122.5k, respectively, primarily due to the input projection. Replacing pose with EmotionCLIP input raises the indicator-context model to 163.7k parameters, entirely through the first convolution, whose input grows from 44 to 513 channels; the frozen EmotionCLIP encoder itself is not trained. The optional face branch introduces another 29.3k parameters, largely from mapping the 768-dimensional face embedding to 32 dimensions.

We optimise all variants using AdamW with learning rate $3\times10^{-4}$, weight decay 0.05, and batch size 16. Training lasts at most 60 epochs for indicator and CLIP contexts or 150 epochs for VLM context, with early stopping after 10 epochs without validation improvement. The selected checkpoint minimises validation cross-entropy in the vote-prediction setting and validation mean absolute error in regression. To augment each training pose sequence, we apply random rotations within $\pm15^\circ$, scale factors in $[0.9,1.1]$, and Gaussian coordinate noise of standard deviation $\sigma=0.02$ shoulder widths. Horizontal reflection is applied with probability 0.5, reversing the horizontal coordinate and exchanging left and right joints. All regions of a clip receive the same transform, and missing joints remain unchanged. These transforms act on joint coordinates, so they apply only to pose input; EmotionCLIP embeddings are extracted once from fixed crops and used without augmentation. For final evaluation, we ensemble five independently seeded models, averaging their prediction scores and using a majority vote of their binary decisions.

\begin{table}[h]
\centering
\caption{Hyper-parameters of ContextPose.}
\label{tab:hparams}
\small
\begin{tabular}{@{}llr@{}}
\toprule
Component & Hyper-parameter & Value \\
\midrule
Input & Steps per window $T$ / joints $J$ / features $F$ & 32 / 11 / 4 \\
& Behaviour input width $D_x$ (pose / EmotionCLIP) & 44 / 513 \\
& Context dimension $d_c$ (indicators / CLIP / VLM) & 4 / 768 / 384 \\
Behaviour encoder & Conv layers / kernel / activation & 2 / 3 / GELU \\
Model width & $W$ & 64 \\
Cross-attention & Heads $H$ / head width $d_h$ & 4 / 16 \\
Region Transformer & Layers / heads / FFN width & 1 / 4 / $2W$ \\
Face stream & Input $D$ / encoding width & 768 / 32 \\
Head & Hidden width & $W/2$ \\
Regularisation & Transformer dropout / head dropout $p_h$ / face dropout $p_f$ & 0.3 / 0.3 / 0.3 \\
\midrule
Optimisation & Optimiser / learning rate / weight decay & AdamW / $3{\times}10^{-4}$ / 0.05 \\
& Batch size / max.\ epochs / early-stopping patience & 16 / 60 / 10 \\
& Ensemble size (seeds) & 5 \\
\midrule
Parameters & Pose input, without / with face stream (indicator context) & 73.6k / 102.9k \\
& EmotionCLIP input, without / with face stream (indicator context) & 163.7k / 193.0k \\
\bottomrule
\end{tabular}
\end{table}

\end{document}
</pasted_content id="3ab9">

What I have so far. simplify some of the complex equations

---


**Claude**

I simplified the denser equations in [contextpose_methodology.tex](reports/contextpose_methodology.tex), starting from the version you pasted. Each detail I took out of an equation now appears in a sentence next to it, so nothing about the model is lost. There's still no LaTeX compiler on this machine; a script check of braces, environments and references found no problems.

| Section | Before | After |
|---|---|---|
| Temporal encoder | Nested conv equation, a separate convolution sum $\sum_k \mW_k\mathbf{x}_{t+k}$, and a boundary condition | One line, $\mZ_r=\GELU(\mathrm{Conv}_2(\GELU(\mathrm{Conv}_1(\mX_r))))$. "Kernel 3, each step depends on itself and its two neighbours" is now said in words. |
| Cross-attention | Separate equations for Q, K, V, the weights $\bm\alpha$, and the multi-head concatenation | One single-head equation: the weights $\bm\alpha_r$ and the summary $\vs_r=\sum_t \alpha_{r,t}\,\mathbf{z}_{r,t}\mW^V$. The 4 heads and the output projection $\mW^O$ are described in text. |
| Inter-region Transformer | Two equations with dropout terms, plus a separate FFN definition | $\mS'=\LN(\mS+\mathrm{MHSA}(\mS))$ and $\mS''=\LN(\mS'+\mathrm{FFN}(\mS'))$ on one line. FFN width, ReLU and dropout are in text. |
| Prediction head | $\mW_4\GELU(\mW_3\Drop_{p_h}(\vh)+\vb_3)+\vb_4$ | $\vy_{\text{beh}}=\mathrm{MLP}(\vh)=\mW_4\GELU(\mW_3\vh+\vb_3)+\vb_4$. Input dropout is mentioned in text. |
| Face stream | Two separate equations, plus an indicator function and a nested case split by $k$ | Projection and masked mean share one display. The correction is now a two-case equation: "$\psi(\ldots)$ if the clip has a valid face step, otherwise $\mathbf{0}$". The $k=1$ (valence only) versus $k=2$ choice is one sentence. Face dropout is now described in words instead of the $\xi\sim\mathrm{Bernoulli}$ notation. |
| Ablations | $\mathrm{MLP}(\Drop_{p_h}(\vchat))$ | $\mathrm{MLP}(\vchat)$ |

I left the loss, the log-odds offset, the residual output and the region average alone, because they're already short.

Your "Pose input" paragraph is still just a heading with no text. I left it that way in case you're writing it yourself. If you want, I can fill it with a two-sentence summary (11 joints × 4 features, so $D_x=44$).

---


**User**

<pasted_content id="3ab9">
\documentclass[11pt]{article}

\usepackage[margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc}
\usepackage{caption}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}

\captionsetup{font=small,labelfont=bf}

\tikzset{
  box/.style={draw,rounded corners=2pt,align=center,font=\footnotesize,minimum height=7mm,inner sep=3pt,fill=white},
  inp/.style={box,fill=blue!6},
  op/.style={box,fill=gray!8},
  outputnode/.style={box,fill=green!8},
  arr/.style={-{Stealth[length=2mm]},thick},
}

% Notation
\newcommand{\R}{\mathbb{R}}
\newcommand{\regions}{\mathcal{R}}
\newcommand{\vc}{\mathbf{c}}
\newcommand{\vchat}{\hat{\mathbf{c}}}
\newcommand{\ve}{\mathbf{e}}
\newcommand{\vq}{\mathbf{q}}
\newcommand{\vs}{\mathbf{s}}
\newcommand{\vh}{\mathbf{h}}
\newcommand{\vy}{\hat{\mathbf{y}}}
\newcommand{\vb}{\mathbf{b}}
\newcommand{\vu}{\mathbf{u}}
\newcommand{\mZ}{\mathbf{Z}}
\newcommand{\mX}{\mathbf{X}}
\newcommand{\mS}{\mathbf{S}}
\newcommand{\mW}{\mathbf{W}}
\newcommand{\GELU}{\operatorname{GELU}}
\newcommand{\ReLU}{\operatorname{ReLU}}
\newcommand{\LN}{\operatorname{LN}}
\newcommand{\Drop}{\operatorname{Drop}}
\newcommand{\softmax}{\operatorname{softmax}}

\title{ContextPose}
\author{}
\date{}

\begin{document}
\maketitle

\section{Methodology}
\label{sec:method}

ContextPose estimates the valence and arousal reflected in human annotations of short video clips depicting a child before, during, and after an environmental event. The proposed architecture is motivated by the hypothesis that contextual information can help identify the behavioural moments most relevant to an affective response. Rather than merely concatenating context with an aggregated behaviour feature, we use the context representation as an attention query over the temporal behaviour features. In this formulation, context determines how the behaviour sequence is summarised, while the attended values themselves are derived from the child's observed behaviour.

The framework is illustrated in Figure~\ref{fig:arch}. The child's behaviour is described by one of two interchangeable inputs: body-pose features computed from tracked joints, or subject-aware frame embeddings computed directly from the video by EmotionCLIP. Whichever input is used, the sequences from the pre-event, event, and post-event intervals are first encoded independently using a shared temporal convolutional network. Event context, represented either through annotations or a language-based description, is mapped into the same latent space. For each interval, the context embedding is combined with a learned region embedding to query the corresponding temporal behaviour tokens using multi-head cross-attention. The resulting three summaries are integrated by a single-layer Transformer and passed to a prediction head for valence and arousal. When reliable facial observations are available, an auxiliary face branch provides an additive correction.

%-----------------------------------------------------------------------------
\begin{figure}[t]
\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[every node/.style={text width=1.75cm}]
  % Behaviour columns (pose or EmotionCLIP input)
  \foreach \name/\x/\lbl in {pre/0/pre-event, ev/3/event, post/6/post-event}{
    \node[inp,text width=2.2cm] (P\name) at (\x,0.15) {Pose or EmotionCLIP\\[-1pt]{\scriptsize $\mX_{\text{\lbl}}\in\R^{T\times D_x}$}};
    \node[op]  (T\name) at (\x,-1.4) {Temporal conv.\\[-1pt]{\scriptsize $\to \mZ_r\in\R^{T\times W}$}};
    \node[op,minimum height=11mm] (A\name) at (\x,-2.9) {Cross-attn.\\[-1pt]{\scriptsize $\vq_r=\vchat+\ve_r$}};
    \draw[arr] (P\name) -- (T\name);
    \draw[arr] (T\name) -- node[right,font=\scriptsize]{$\mZ_r$} (A\name);
  }

  % Context path
  \node[inp,text width=2.3cm] (C) at (-2.9,1.35) {Context $\vc$\\[-1pt]{\scriptsize indicators / CLIP / VLM}};
  \node[op] (Wc) at (-0.15,1.35) {Linear\\[-1pt]{\scriptsize $\vchat=\mW_c\vc+\vb_c$}};
  \draw[arr] (C) -- (Wc);
  \draw[thick] (Wc.east) -- (7.4,1.35);
  \foreach \name/\x in {pre/1.4, ev/4.4, post/7.4}{
    \draw[arr] (\x,1.35) |- (A\name.east);
  }

  % Region transformer and head
  \node[op,text width=7.6cm] (RT) at (3,-4.6) {Inter-region Transformer over $[\vs_{\text{pre}};\vs_{\text{event}};\vs_{\text{post}}]$, then mean over regions $\to \vh$};
  \foreach \name in {pre,ev,post}{ \draw[arr] (A\name) -- node[right,font=\scriptsize]{$\vs_r$} (A\name |- RT.north); }
  \node[op,text width=3.2cm] (H) at (3,-5.85) {MLP head $W\!\to\!W/2\!\to\!2$};
  \draw[arr] (RT) -- (H);
  \node[draw,circle,inner sep=1pt,text width=0.4cm,font=\small] (plus) at (3,-7.0) {$+$};
  \draw[arr] (H) -- node[right,font=\scriptsize]{$\vy_{\text{beh}}$} (plus);
  \node[inp,text width=2.6cm] (bg) at (-0.2,-7.0) {Offset $\vb_{g(\vc)}$\\[-1pt]{\scriptsize context lookup}};
  \draw[arr] (bg) -- (plus);
  \node[outputnode,text width=2.6cm] (y) at (3,-8.1) {$\vy=[\hat y_{\text{val}},\hat y_{\text{aro}}]$};
  \draw[arr] (plus) -- (y);

  % Face stream
  \node[inp,text width=2.4cm] (F) at (9.7,0) {Face features $\mathbf{x}^{f}_{r,t}$, gate $m_{r,t}$};
  \node[op,text width=2.4cm] (FE) at (9.7,-1.4) {LN $\to$ Linear $\to$ GELU\\[-1pt]{\scriptsize $\vu_{r,t}\in\R^{32}$}};
  \node[op,text width=2.4cm,minimum height=11mm] (FP) at (9.7,-2.9) {Gated mean per region\\[-1pt]{\scriptsize $\bar{\vu}_r$}};
  \node[op,text width=2.4cm] (FH) at (9.7,-4.6) {Face head $\psi$\\[-1pt]{\scriptsize zero-initialised}};
  \draw[arr] (F) -- (FE);
  \draw[arr] (FE) -- (FP);
  \draw[arr] (FP) -- (FH);
  \draw[arr] (FH.south) |- node[pos=0.75,above,font=\scriptsize]{$\bm{\Delta}_{\text{face}}$} (plus.east);
\end{tikzpicture}}
\caption{ContextPose. Each window's behaviour sequence, given either as pose features or as subject-aware EmotionCLIP frame embeddings, is encoded by a temporal convolution, and a query formed from the projected context and a learned region embedding attends over the window's time steps; the convolution and the cross-attention share their weights across the three windows. A Transformer layer relates the three window summaries, which are then averaged and decoded to valence and arousal. An optional gated face stream adds a zero-initialised correction, and the output is a residual on a context-lookup offset.}
\label{fig:arch}
\end{figure}
%-----------------------------------------------------------------------------

\subsection{Problem formulation and inputs}
\label{sec:inputs}

We organise each clip into three event-centred intervals, indexed by $r\in\regions=\{\text{pre},\text{event},\text{post}\}$: a 5\,s pre-event interval, the event interval itself, and a 5\,s post-event interval. Clips lasting 10\,s or less are omitted because they contain no event frames under this partition. Each interval is resampled to $T=32$ temporal steps, and its behaviour input is a sequence of step vectors $\mX_r\in\R^{T\times D_x}$, built either from pose or from EmotionCLIP as described below. The overall clip is associated with an event-context vector $\vc\in\R^{d_c}$. If the facial branch is active, each interval additionally provides face embeddings $\mathbf{X}^f_r\in\R^{T\times D}$ and a binary validity mask $\mathbf{m}_r\in\{0,1\}^{T}$. The network outputs two unconstrained values $\vy\in\R^2$, corresponding to valence and arousal, whose interpretation is specified by the selected learning objective.

% \paragraph{Pose representation.}
 % \We retain $J=11$ upper-body COCO landmarks, comprising the nose, eyes, ears, shoulders, elbows, and wrists, extracted at $f_s=25$\,fps. Let $\mathbf{p}_{n,j}\in\R^2$ denote the image coordinates of joint $j$ in frame $n$. To reduce sensitivity to camera placement and apparent subject scale, coordinates are centred at the shoulder midpoint $\mathbf{o}_n$ and normalised by the clip-level median shoulder width $w$:

% \begin{equation}
%   \tilde{\mathbf{p}}_{n,j} = \frac{\mathbf{p}_{n,j}-\mathbf{o}_n}{w},
% \end{equation}

% This transformation suppresses differences in image position and apparent camera distance while retaining relative body configuration. To obtain fixed-size inputs from intervals of unequal duration, we resample each interval to $T=32$ temporal steps. Each joint is described by $F=4$ features: two normalised coordinates, detector confidence restricted to $[0,1]$, and movement speed. Missing detections are assigned zero coordinates and confidence. During interpolation, a step uses its two surrounding observations when both exist; if only one exists, that observation is retained, and if neither exists, the step remains zero. This avoids inventing trajectories across missing observations, with the confidence feature explicitly indicating uncertainty.

% Fixed-length resampling alone does not preserve movement rate: an event lasting 20\,s and a surrounding interval lasting 5\,s would both contain 32 steps despite their different physical durations. We therefore calculate joint speed from the original frame sequence, before temporal resampling, and express it in shoulder widths per second:

% \begin{equation}
%   v_{n,j} = \min\!\left(f_s\,\lVert\tilde{\mathbf{p}}_{n,j}-\tilde{\mathbf{p}}_{n-1,j}\rVert_2,\; v_{\max}\right), \qquad v_{\max}=20.
% \end{equation}

% If a valid previous observation is unavailable, the next observed frame is used; joints lacking a valid neighbouring observation receive zero speed. The limit $v_{\max}$ reduces sensitivity to abrupt tracking errors. For each resampled step, the speed feature is the average over observed frames in a centred span of approximately $N_r/T$ original frames, where $N_r$ is the interval length. Accordingly, the feature retains a consistent real-time interpretation across intervals.

\paragraph{Pose input.}


\paragraph{Subject-aware video input.} Alternatively, each step is described directly from the video by EmotionCLIP (Zhang et al., 2023), a CLIP ViT-B/32 image encoder trained on vlog videos and their transcripts to represent the emotional state of a designated person. EmotionCLIP is subject-aware: alongside the RGB frame, it receives a binary mask marking the person to be read, which is average-pooled over each $32\times32$ patch and added to that patch's positional embedding. The encoder therefore attends to the subject's body and face while still seeing the surrounding scene. At each of the $T$ steps, we take the corresponding video frame, crop a square centred on the child with side 0.875 times the frame's shorter side, and resize it to 224\,px with ImageNet normalisation. EmotionCLIP was trained with person boxes from an object detector; we instead build the mask from the child's tracked joints, as their bounding box padded by 0.3 shoulder widths and extended downwards, because the 11 joints cover mainly the upper body. Deriving the mask from the child's own track keeps it on the child rather than on an adult or a person shown on a screen. When the child is not tracked within two frames of the step, the mask is left empty rather than guessed. The frozen encoder's output is $\ell_2$-normalised to a 512-dimensional frame embedding, and a binary flag recording whether the child was tracked is appended, giving $D_x=513$. Like the confidence feature of the pose input, this flag tells the encoder which steps are less reliable. We use EmotionCLIP's per-frame encoder rather than its own temporal Transformer, so that our temporal encoder and context-queried attention operate on the same $T$ steps for both inputs. With this input, pose only determines where the crop and mask are placed; joint coordinates do not enter the features.

\subsection{Temporal behaviour encoder}
\label{sec:pose-encoder}

For each region, the behaviour input $\mX_r\in\R^{T\times D_x}$ holds one vector per step: the concatenated joint features for pose, or the flagged frame embedding for EmotionCLIP. Two one-dimensional temporal convolutions turn this sequence into $W$-dimensional tokens:

\begin{equation}
  \mZ_r = \GELU\!\big(\mathrm{Conv}_2(\GELU(\mathrm{Conv}_1(\mX_r)))\big)\in\R^{T\times W}.
\end{equation}

Each convolution has kernel size 3 with zero padding, so every output step is a learned linear function of the input at that step and its two neighbours. $\mathrm{Conv}_1$ maps $D_x$ input channels to $W$, and $\mathrm{Conv}_2$ maps $W$ to $W$. Only $\mathrm{Conv}_1$ depends on the choice of input; every later layer is identical for pose and EmotionCLIP. Stacking the two layers yields a five-step receptive field, enabling each token $\mathbf{z}_{r,t}$ to encode local change across neighbouring steps rather than a single isolated observation. Parameters are shared across the three regions. No explicit temporal positional embedding is introduced: ordering is represented locally through convolution, while the subsequent within-region attention is otherwise permutation-invariant over the generated tokens.

\subsection{Context encoder}
\label{sec:context}

We investigate three alternative representations of environmental context, each providing a different level of semantic detail. The first consists of four annotator-defined binary indicators describing social interaction, auditory events, activity or object changes, and visual or media events ($d_c=4$); clips without annotated events receive an all-zero vector. The second representation derives descriptions from the video itself. Qwen3-Omni generates accounts of the scene, the child's behaviour, and audible surroundings over successive 5--6\,s segments. These descriptions are encoded using all-MiniLM-L6-v2, averaged across segments, and $\ell_2$-normalised to obtain a clip-level vector ($d_c=384$). Each representation is mapped to the common latent width using a linear projection:

\begin{equation}
  \vchat = \mW_c\,\vc + \vb_c\in\R^{W}.
\end{equation}

In addition, each temporal region is assigned a learned embedding $\ve_r\in\R^W$, with parameters initialised from $\mathcal{N}(0,0.02^2)$. Because the cross-attention module shares parameters across regions, these embeddings provide the region-specific identity needed to distinguish pre-event, event, and post-event queries.

\subsection{Context-queried temporal attention}
\label{sec:cross-attention}

The central fusion mechanism is context-conditioned temporal cross-attention. For each region, the model constructs a single query from the projected environmental context and the corresponding region embedding, $\vq_r = \vchat + \ve_r$; keys and values come exclusively from that region's behaviour tokens. For one attention head, the weights over the $T$ steps and the resulting summary are

\begin{equation}
  \bm{\alpha}_r = \softmax\!\left(\frac{(\vq_r\mW^{Q})(\mZ_r\mW^{K})^{\top}}{\sqrt{d_h}}\right)\in\R^{T},
  \qquad
  \vs_r = \sum_{t=1}^{T}\alpha_{r,t}\,\mathbf{z}_{r,t}\mW^{V}.
  \label{eq:attn}
\end{equation}

We use $H=4$ heads, each with its own projections $\mW^{Q},\mW^{K},\mW^{V}\in\R^{W\times d_h}$ and width $d_h=W/H$. The head outputs are concatenated and mixed by an output projection $\mW^{O}\in\R^{W\times W}$ to give $\vs_r\in\R^{W}$; biases are omitted from the notation. The same attention projections are reused across all three regions.

As shown by Equation~\eqref{eq:attn}, environmental context influences the attention distribution rather than being inserted directly into the behaviour-derived values. For each head, the region summary is a weighted average of projected behaviour tokens, and the context vector $\vc$ affects which tokens receive the greatest weights. Consequently, an auditory event and a social interaction may lead the model to emphasise different periods of the same behavioural sequence. This mechanism produces a fixed-dimensional summary $\vs_r$ for every region, independent of its original temporal extent.

\subsection{Inter-region Transformer and prediction head}
\label{sec:region-transformer}

The regional summaries are assembled into $\mS=[\vs_{\text{pre}};\vs_{\text{event}};\vs_{\text{post}}]\in\R^{3\times W}$. A single post-normalisation Transformer encoder layer models relationships among these representations:

\begin{equation}
  \mS' = \LN\big(\mS + \mathrm{MHSA}(\mS)\big),
  \qquad
  \mS'' = \LN\big(\mS' + \mathrm{FFN}(\mS')\big).
\end{equation}

Here $\mathrm{MHSA}$ is self-attention over the three region vectors with $H=4$ heads, and $\mathrm{FFN}$ is a two-layer network of width $2W$ with a ReLU activation. Dropout, omitted from the equation, is applied to the attention weights, inside the feed-forward network, and to the output of each sublayer. Through this operation, the representation of a given region can incorporate information from the others. For example, behaviour during the event can be interpreted in relation to the child's behaviour immediately before and afterward. We obtain a clip-level representation by averaging the transformed regional outputs:

\begin{equation}
  \vh = \frac{1}{3}\sum_{r\in\regions}\mS''_r\in\R^{W}.
\end{equation}

We do not apply positional encodings at this stage. Since self-attention followed by mean aggregation is invariant to permutations of the three input rows, the ordering of the regional vectors is not explicitly available to the final representation. The learned region embeddings introduced at the preceding attention stage are therefore the only source of region identity.

An optional \emph{context skip} connection modifies the fused representation as $\vh\leftarrow\vh+\vchat$, allowing context to influence the output directly as well as through temporal selection. A two-layer MLP with an intermediate dimension of $W/2$ maps the resulting representation to the behaviour-branch valence and arousal scores:

\begin{equation}
  \vy_{\text{beh}} = \mathrm{MLP}(\vh) = \mW_4\,\GELU(\mW_3\vh+\vb_3)+\vb_4\in\R^{2},
\end{equation}

with dropout of rate $p_h$ applied to its input during training.

\subsection{Gated face stream}
\label{sec:face}

Although facial expression can provide strong evidence of valence, its reliability varies considerably in naturalistic recordings of children. Face crops may be occluded, blurred, poorly illuminated, or affected by head rotation. We therefore model facial information through a separate auxiliary branch that contributes an additive correction to the behaviour-based estimate only when adequate facial observations are available.

For each of the $T$ temporal positions in a region, we extract the face crop from the nearest video frame. The crop is centred using the tracked facial landmarks, covers a square region of width 0.9 shoulder widths, and is resized to 224\,px. A frozen facial-expression ViT pretrained on FER2013 produces a $D=768$-dimensional [CLS] embedding $\mathbf{x}^f_{r,t}\in\R^{D}$. A binary mask $m_{r,t}$ identifies valid crops according to three criteria: (i) the pose tracker detects the nose and both eyes, indicating a frontal face; (ii) YuNet independently detects a face at that location with confidence at least 0.8; and (iii) the crop exceeds a mean grey-level threshold of 40 and a Laplacian-variance threshold of 10. These checks reject unreliable detections, including cases in which the pose estimator reports facial landmarks despite occlusion or a lowered head. Invalid steps have their features set to zero.

Each face embedding is projected to 32 dimensions, and the valid steps of each region are averaged:

\begin{equation}
  \vu_{r,t} = \GELU\big(\mW_f\,\LN(\mathbf{x}^f_{r,t})+\vb_f\big)\in\R^{32},
  \qquad
  \bar{\vu}_r = \frac{\sum_{t} m_{r,t}\,\vu_{r,t}}{\max\big(1,\sum_{t} m_{r,t}\big)}.
\end{equation}

The denominator $\max(1,\cdot)$ ensures that a region with no valid facial frames has a zero summary. The three region summaries are concatenated and passed to a two-layer head $\psi$ ($96\to W/2\to k$) with input dropout $p_h$, which produces the correction

\begin{equation}
  \bm{\Delta}_{\text{face}} =
  \begin{cases}
    \psi\big([\bar{\vu}_{\text{pre}};\bar{\vu}_{\text{event}};\bar{\vu}_{\text{post}}]\big), & \text{if the clip has at least one valid face step},\\
    \mathbf{0}, & \text{otherwise}.
  \end{cases}
  \label{eq:face}
\end{equation}

With $k=2$, the face stream corrects both valence and arousal; with $k=1$, it corrects valence only and the arousal entry of $\bm{\Delta}_{\text{face}}$ is zero.

We introduce three safeguards to limit the effect of low-quality facial information. First, the final layer of $\psi$ is initialised to zero, so the model initially reproduces the behaviour-and-context prediction and only later learns an additional facial contribution. Second, the second case of Equation~\eqref{eq:face} forces $\bm{\Delta}_{\text{face}}=\mathbf{0}$ when no valid facial step exists anywhere in the clip, preventing a spurious correction from the head's bias. Third, during training, the complete face branch is randomly removed with probability $p_f$ by setting every mask entry $m_{r,t}$ of the clip to zero. This face dropout also makes the correction zero and encourages the remaining modalities to support the prediction independently.

\subsection{Residual prediction on a context lookup}
\label{sec:residual}

Because event-indicator combinations are informative about the observed annotation distribution, a simple context-conditioned lookup provides a useful baseline. Given the small development sample of approximately 150 clips, we train ContextPose to estimate deviations from this baseline rather than requiring its neural branches to learn the full prediction directly:

\begin{equation}
  \vy = \vy_{\text{beh}} + \bm{\Delta}_{\text{face}} + \vb_{g(\vc)}.
  \label{eq:output}
\end{equation}

In Equation~\eqref{eq:output}, $g(\vc)$ identifies the event-indicator combination associated with the clip, and $\vb_{g}\in\R^2$ denotes its fixed baseline offset. Offsets are estimated exclusively from training examples within each cross-validation fold and are not updated by gradient descent, thereby excluding evaluation data from baseline estimation. For vote-based learning, each offset equals the log-odds of the group's mean vote proportion $\bar{\pi}_{g}$. We clip this proportion to $[\epsilon,1-\epsilon]$, where $\epsilon=0.02$, to avoid infinite log-odds:

\begin{equation}
  \vb_{g} = \log\frac{\bar{\pi}_{g}}{1-\bar{\pi}_{g}}.
\end{equation}

For regression, the offset is instead the median target value for the corresponding group. If fewer than three training clips are available for a group, the global training-set estimate is used. Disabling the lookup sets the offset to zero; the prediction head's bias then supplies any learned constant component, and no label statistics are injected through this branch.

\subsection{Output parameterisation and objective}
\label{sec:objective}

Three annotators assessed each clip, assigning positive, neutral, or negative valence and either high or low arousal. Given the moderate inter-annotator agreement, our primary objective retains the distribution of votes instead of collapsing them to a single consensus class. For each dimension, $\pi\in[0,1]$ denotes the proportion of known votes for positive valence or high arousal. The prediction $\hat{y}$ is interpreted as the logit of that vote probability, and training minimises binary cross-entropy with the soft vote proportion for both outputs, averaged over the batch:

\begin{equation}
  \mathcal{L} = -\sum_{k\in\{\text{val},\,\text{aro}\}}\Big[\pi_k\log\sigma(\hat{y}_k)+(1-\pi_k)\log\big(1-\sigma(\hat{y}_k)\big)\Big].
\end{equation}

During inference, $\sigma(\hat{y}_k)$ gives the estimated vote probability, used to distinguish positive from non-positive valence or high from low arousal after applying a threshold. Thresholds are chosen to maximise balanced accuracy using only model-fitting and selection data, never evaluation clips. For the alternative regression objective, the two outputs represent consensus valence and arousal directly and are optimised using mean squared error.

\subsection{Stream ablations}
\label{sec:ablations}

To assess the contribution of each modality, we evaluate stream-removal variants that retain the same basic prediction framework while disabling selected inputs:

\begin{align}
  \text{behaviour only:}\quad & \vy = \vy_{\text{beh}}\big|_{\vchat=\mathbf{0}} + \bm{\Delta}_{\text{face}} + \vb, \\
  \text{context only:}\quad & \vy = \mathrm{MLP}(\vchat) + \bm{\Delta}_{\text{face}} + \vb, \\
  \text{face only:}\quad & \vy = \bm{\Delta}_{\text{face}} + \vb.
\end{align}

In the behaviour-only variant, the projected context is set to zero, leaving the learned region embeddings to provide the cross-attention queries; it can be run with either pose or EmotionCLIP input. In the context-only variant, the behaviour pathway is bypassed and the prediction MLP operates directly on the projected context. The face-only configuration removes the behaviour prediction and retains the facial correction where that branch is enabled. Across these experiments, $\vb$ denotes either the fixed context-dependent offset $\vb_{g(\vc)}$ or zero when the lookup is inactive. Thus, when the lookup is enabled, the nominal face-only condition still incorporates event information through its baseline offset.

\subsection{Implementation details}
\label{sec:impl}

We keep the network compact to limit overfitting on the development set of approximately 150 clips. Table~\ref{tab:hparams} summarises the architectural and optimisation settings. With latent width $W=64$, pose input, and the indicator context, the model contains 73.6k trainable parameters; using VLM or CLIP features increases this to 98.0k and 122.5k, respectively, primarily due to the input projection. Replacing pose with EmotionCLIP input raises the indicator-context model to 163.7k parameters, entirely through the first convolution, whose input grows from 44 to 513 channels; the frozen EmotionCLIP encoder itself is not trained. The optional face branch introduces another 29.3k parameters, largely from mapping the 768-dimensional face embedding to 32 dimensions.

We optimise all variants using AdamW with learning rate $3\times10^{-4}$, weight decay 0.05, and batch size 16. Training lasts at most 60 epochs for indicator and CLIP contexts or 150 epochs for VLM context, with early stopping after 10 epochs without validation improvement. The selected checkpoint minimises validation cross-entropy in the vote-prediction setting and validation mean absolute error in regression. To augment each training pose sequence, we apply random rotations within $\pm15^\circ$, scale factors in $[0.9,1.1]$, and Gaussian coordinate noise of standard deviation $\sigma=0.02$ shoulder widths. Horizontal reflection is applied with probability 0.5, reversing the horizontal coordinate and exchanging left and right joints. All regions of a clip receive the same transform, and missing joints remain unchanged. These transforms act on joint coordinates, so they apply only to pose input; EmotionCLIP embeddings are extracted once from fixed crops and used without augmentation. For final evaluation, we ensemble five independently seeded models, averaging their prediction scores and using a majority vote of their binary decisions.

\begin{table}[h]
\centering
\caption{Hyper-parameters of ContextPose.}
\label{tab:hparams}
\small
\begin{tabular}{@{}llr@{}}
\toprule
Component & Hyper-parameter & Value \\
\midrule
Input & Steps per window $T$ / joints $J$ / features $F$ & 32 / 11 / 4 \\
& Behaviour input width $D_x$ (pose / EmotionCLIP) & 44 / 513 \\
& Context dimension $d_c$ (indicators / CLIP / VLM) & 4 / 768 / 384 \\
Behaviour encoder & Conv layers / kernel / activation & 2 / 3 / GELU \\
Model width & $W$ & 64 \\
Cross-attention & Heads $H$ / head width $d_h$ & 4 / 16 \\
Region Transformer & Layers / heads / FFN width & 1 / 4 / $2W$ \\
Face stream & Input $D$ / encoding width & 768 / 32 \\
Head & Hidden width & $W/2$ \\
Regularisation & Transformer dropout / head dropout $p_h$ / face dropout $p_f$ & 0.3 / 0.3 / 0.3 \\
\midrule
Optimisation & Optimiser / learning rate / weight decay & AdamW / $3{\times}10^{-4}$ / 0.05 \\
& Batch size / max.\ epochs / early-stopping patience & 16 / 60 / 10 \\
& Ensemble size (seeds) & 5 \\
\midrule
Parameters & Pose input, without / with face stream (indicator context) & 73.6k / 102.9k \\
& EmotionCLIP input, without / with face stream (indicator context) & 163.7k / 193.0k \\
\bottomrule
\end{tabular}
\end{table}

\end{document}
</pasted_content id="3ab9">

This is what I have so far.
Generate a clear but not cluttered image diagram of the architecture. take your time to do it properly

---


**User**

[Image: original 2007x1559, displayed at 2000x1554. Multiply coordinates by 1.00 to map to original image.]

---


**Claude**

The first draft renders cleanly. I'm now fixing the problems I can see in it: text overflowing in the face-stream boxes, the right edge being clipped, and the routing into the lookup box.

---


**User**

[Image: original 2019x1559, displayed at 2000x1544. Multiply coordinates by 1.01 to map to original image.]

---


**Claude**

I've made the architecture diagram, rendered it, and checked it by eye; a second pass fixed overflowing text and a clipped edge. I haven't compiled it into your LaTeX document yet. It's in `reports/figures/`:

- [contextpose_architecture.pdf](reports/figures/contextpose_architecture.pdf): vector version, best for the paper
- [contextpose_architecture.svg](reports/figures/contextpose_architecture.svg) and a 300 dpi PNG
- [contextpose_architecture.py](reports/figures/contextpose_architecture.py): the script that draws it, so you can change labels or colours

**Layout and colours.** It reads top to bottom:
- **Blue (behaviour):** the clip timeline, then the input features, shown as either pose or EmotionCLIP with an "or" between them.
- **Purple (fusion):** the cross-attention, the inter-region Transformer and the MLP head.
- **Orange (context):** comes in from the left.
- **Green (face):** the optional face stream, on the right inside a dashed box.
- **Grey:** frozen pretrained models.

The cross-attention box has small bar charts showing different attention patterns for each window. They illustrate the idea that the context chooses which time steps count, and they don't show real attention weights. The legend explains frozen, trainable, fixed and optional.

**Left out on purpose.** The optional context skip ($\vh+\vchat$) isn't drawn, to keep the figure uncluttered. Mention it in the caption if you need it.

**To use it,** replace the TikZ figure with this. The path assumes the `.tex` file is in `reports/`. If you remove the `tikz` package, add `\usepackage{graphicx}`.

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/contextpose_architecture.pdf}
\caption{ContextPose. Each window's behaviour sequence (pose features or subject-aware EmotionCLIP embeddings) is encoded by a shared temporal convolution. A query formed from the projected context and a learned region embedding then attends over the window's time steps. A Transformer layer relates the three window summaries, which are averaged and decoded to valence and arousal. The prediction is a residual on a fixed context-lookup offset, and an optional gated face stream adds a zero-initialised correction.}
\label{fig:arch}
\end{figure}
```

To regenerate it after editing the script (matplotlib runs in a temporary environment, so the project isn't touched):

```bash
uv run --no-project --with matplotlib python -I reports/figures/contextpose_architecture.py
```

**CLIP is still mentioned in your draft.** You've dropped the CLIP context from the Context encoder section, and the diagram shows only indicators or VLM. CLIP still appears in three places:
- the TikZ context box ("indicators / CLIP / VLM");
- the implementation paragraph ("using VLM or CLIP features … 122.5k" and "60 epochs for indicator and CLIP contexts");
- the table row "indicators / CLIP / VLM: 4 / 768 / 384".

---


**User**

<bash-input>uv run --no-project --with matplotlib python -I reports/figures/contextpose_architecture.py</bash-input><bash-stdout>'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
wrote /Users/henryansah/workspace/cvpr/reports/figures/contextpose_architecture.pdf</bash-stdout><bash-stderr></bash-stderr>

---


**User**

This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

<system-reminder>
<tools-used-before-summary>Bash, Edit, Read, SendUserFile, Write, mcp__ccd_turn__update_plan, mcp__ccd_turn__wrap_up</tools-used-before-summary>
</system-reminder>
Summary:
1. **Primary Request and Intent:**
   - **Earlier (already done):**
     - Rewrite the ContextPose methodology as paragraphs, checked against the code.
     - Remove FiLM.
     - Remove `\S\ref`.
     - Explain the sentence about context contributing "no features of its own".
   - **Follow-up 1:** "For parts where we describe joint features as an input stream, joints features can also be replace by subject aware encoder directly from input videos such as EmotionClip … So its either pose or EmotionClip features." Done: the behaviour stream is now either pose or EmotionCLIP.
   - **Follow-up 2:** "What I have so far. simplify some of the complex equations". Done.
   - **Latest:** the user pasted their current version and said: "This is what I have so far. Generate a clear but not cluttered image diagram of the architecture. take your time to do it properly". The diagram is generated, inspected, refined and sent. The closing message still needs to be written.
   - **Constraints:** the user stated no security constraints. System context: the user's email and personal details are private and must not be sent to external services.

2. **Key Technical Concepts:**
   - **Model structure:** ContextPose; behaviour stream input X_r ∈ R^{T×D_x} is either pose (J=11, F=4, D_x=44) or EmotionCLIP (512-d + tracked flag, D_x=513).
   - **EmotionCLIP extraction:**
     - Model: CVPR 2023, Zhang et al.; ViT-B/32 with a subject mask that is average-pooled per 32×32 patch and added to the positional embedding.
     - Crop: 0.875 × the frame's short side, centred on the child, resized to 224 px with ImageNet normalisation.
     - Mask: the joint box padded 0.3 × scale (shoulder width), extended downwards; ±2 frames search; empty when the child is untracked.
     - Output: per-frame encoder, not the temporal transformer; L2-normalised.
     - The earlier E3 network ran it through the face slot, so the pose-slot variant hasn't been run.
   - **Temporal encoder:** 2× Conv1d (k=3, GELU), shared across windows.
   - **Query:** q_r = ĉ + e_r.
   - **Cross-attention:** 4 heads, shared across windows.
   - **Inter-region Transformer:** 1 post-norm layer, then a mean over windows giving h.
   - **Head:** MLP W→W/2→2, giving y_beh.
   - **Face stream (gated):** FER ViT, LN→Linear→GELU to 32 dimensions, masked mean, ψ zero-initialised, Δ_face zero if there is no valid face.
   - **Output:** residual on the context-lookup offset b_g(c); BCE on vote share.
   - **Parameters (W=64):**

     | Configuration | Parameters |
     |---|---|
     | Pose | 73.6k |
     | Pose + face | 102.9k |
     | EmotionCLIP | 163.7k |
     | EmotionCLIP + face | 193.0k |
     | VLM context | 98.0k |
     | CLIP context | 122.5k |

   - **Diagram tooling:** matplotlib 3.11.2 via `uv run --no-project --with matplotlib`. Text in Arial; math via mathtext with the cm fontset; pdf.fonttype 42. rsvg-convert is available.

3. **Files and Code Sections:**
   - **/Users/henryansah/workspace/cvpr/reports/contextpose_methodology.tex**
     - Holds my "simplified equations" version, which includes the CLIP context sentence.
     - The user's latest paste differs: the context encoder lists only indicators and VLM, the "Pose input" paragraph is empty, and the intro's last sentence is removed.
     - Backup of the older draft: `/private/tmp/claude-501/-Users-henryansah--claude-spares-spare-5WONKs/67cd9486-2e25-4f58-9bb8-bc406f97875a/scratchpad/contextpose_methodology.before_eclip.tex`
   - **/Users/henryansah/workspace/cvpr/reports/figures/contextpose_architecture.py** (CREATED). This is the diagram generator.
     - Run with: `uv run --no-project --with matplotlib python reports/figures/contextpose_architecture.py`
     - Canvas: 171 × 132 mm. FS=7, FS_S=6.
     - Palette by stream:
       - beh: blue #E4EDF8 / #3A6EA8
       - ctx: orange #FCEBD6 / #C2741F
       - face: green #E2F1E6 / #3B8A57
       - fuse: purple #EDE7F5 / #6A52A0
       - frozen: grey #EFEFEF / #8C8C8C
       - out: white / #2B2B2B
       - fixed: white / #C2741F
     - Geometry:
       - COLS=(62, 90, 118); SPAN_L/R = 47/133; CTX_X=22; FACE_X=152; FW=28.
       - Rows: Y_TIME=124, Y_EXTRACT=111, Y_X=97, Y_ENC=84, Y_ATT=66, Y_REG=47.5, Y_HEAD=36, Y_SUM=26, Y_OUT=14.5.
     - Contents, top to bottom:
       - Clip timeline: pre-event · 5 s | event (variable length) | post-event · 5 s.
       - Grey "Per-step behaviour features" band with two pills: "Pose (11 joints): x, y, confidence, speed → D_x=44" *or* "EmotionCLIP (subject mask): frame embedding + flag → D_x=513". Also the text "T = 32 steps per window".
       - X_pre / X_event / X_post token strips.
       - Wide "Temporal encoder (shared across windows)" box.
       - Wide "Context-queried temporal cross-attention" box with illustrative α bar profiles (a different peak per window) above the Z token strips.
       - Wide "Inter-region Transformer (1 layer, 4 heads)" box.
       - MLP head, then ⊕, then the output box "ŷ = [ŷ_val, ŷ_aro]: logits of the annotators' vote share".
       - Left column:
         - "Event context c: indicators (d_c=4) or VLM description (d_c=384)".
         - Linear (ĉ = W_c c + b_c), then ⊕, joined by the "Region emb. e_r" box.
         - Query arrow labelled "query" and "q_r = ĉ + e_r".
         - Orange elbow line from the context box to the "Context lookup" box ("offset b_g(c), not trained; fit on training folds only"), which feeds ⊕.
       - Right: a dashed group "Optional face stream":
         - FER ViT (grey: face crop at each step, quality gate m_{r,t}), then x^f_{r,t}.
         - Step projection (LN → Linear → GELU), then u_{r,t} ∈ R^32.
         - Gated mean per window, giving ū_r.
         - Face head ψ (zero-initialised).
         - Green line labelled "Δ_face (zero if no valid face)" into ⊕.
       - Legend: frozen, pretrained / trainable / fixed, not trained / optional (dashed).
     - Helper functions: `box(cx, cy, w, h, kind, ls, lw, r, z)` (FancyBboxPatch round); `text(...)`; `label(cx, cy, title, sub)`; `arrow(p, q)` (FancyArrowPatch `-|>,head_length=2.2,head_width=1.1`); `path_arrow(points)`.
     - Saves pdf, svg and png (300 dpi).
   - **Outputs in reports/figures/:**
     - contextpose_architecture.pdf (87 KB)
     - contextpose_architecture.svg (226 KB; validated with rsvg-convert)
     - contextpose_architecture.png (288 KB, 2019×1559)
   - **Files read for facts:**
     - src/asd_contextpose/model.py: ModelConfig, build_model, ContextPose.
     - src/asd_contextpose/face.py
     - reports/asd_contextpose_report.tex: the EmotionCLIP section, around lines 574–612.
     - .waylog/history/2026-11-07_22-06-14Z-claude-what-is-the-context-lookup.md, lines 344–392: EmotionCLIP input details.
     - .waylog/history/2026-11-03_18-46-22Z-…md: extractor details.

4. **Errors and fixes:**
   - **matplotlib missing from .venv.** Ran it with `uv run --no-project --with matplotlib` in an isolated environment instead.
   - **No pdffonts or LaTeX compiler.** Could not check embedded fonts or compile the document; checked the LaTeX with a Python static script instead (braces, environments, undefined references).
   - **First diagram draft problems, all fixed in the second render:**

     | Problem | Fix |
     |---|---|
     | Face box "LN → Linear → GELU" overflowed | Renamed to "Step projection", with the operations as the subtitle |
     | Face group clipped at the right edge | FACE_X=152, FW=28, W_MM=171 |
     | Lookup arrow entered the box off-centre | Elbow route at x=6.8 into the box's left side; box at LOOK_X=26, width 28 |
     | Region-embedding box too close to the lookup line | EMB_X=16.5 |
     | Face projection out of line with its row | Aligned with Y_ENC; x^f and u labels added on the arrows |

   - The user gave no feedback on any errors.

5. **Problem Solving:**
   - The diagram was rendered, inspected visually, refined, and sent to the user with SendUserFile.
   - Inconsistencies in the user's latest document, not yet reported:
     - The implementation paragraph still says "using VLM or CLIP features increases this to 98.0k and 122.5k".
     - It also says "60 epochs for indicator and CLIP contexts".
     - The table's d_c row is "4 / 768 / 384 (indicators / CLIP / VLM)".
     - The TikZ context box says "indicators / CLIP / VLM".
     - The "Pose input" paragraph is empty.

6. **All user messages:**
   - [Original LaTeX paste] "Rewrite the methodology for the Context Pose Architecture. Reduce bullets and use paragraphs to make the info whole. You can make reference from the code."
   - "remove everything FilM from it"
   - "remove these references too: \S\ref"
   - "what does this mean: The context therefore contributes no features of its own to the pose summary; it selects which time steps of each window are summarised"
   - [Paste] "This is what I'm currently working with. For parts where we describe joint features as an input stream, joints features can also be replace by subject aware encoder directly from input videos such as EmotionClip" [interrupted, then repasted] ". So its either pose or EmotionClip features"
   - [Paste] "What I have so far. simplify some of the complex equations"
   - [Paste, with CLIP removed from the context encoder] "This is what I have so far. Generate a clear but not cluttered image diagram of the architecture. take your time to do it properly"

7. **Pending Tasks:**
   - Write the closing message for the diagram task. wrap_up has already been called, and the PNG has already been sent.

8. **Current Work:**
   - The diagram is finished in reports/figures: contextpose_architecture.{py,pdf,svg,png}.
   - Already called:
     - SendUserFile with the PNG, caption "ContextPose architecture diagram. Use the PDF version in the paper."
     - wrap_up with the recap: "The architecture diagram is in reports/figures as PDF, SVG and PNG, with the script that generates it. Your document still uses the TikZ figure. I checked the rendered PNG by eye, but I haven't compiled it inside your LaTeX document."
   - The closing text message to the user was not yet written.

9. **Optional Next Step:**
   - Write the closing message for the request "Generate a clear but not cluttered image diagram of the architecture. take your time to do it properly". It should:
     - Describe what the diagram shows and where the files are, linking [contextpose_architecture.pdf](reports/figures/contextpose_architecture.pdf) and the script.
     - Give the regeneration command: `uv run --no-project --with matplotlib python reports/figures/contextpose_architecture.py`
     - Give the LaTeX snippet to replace the TikZ figure (requires `\usepackage{graphicx}` if TikZ is removed):
       ```latex
       \includegraphics[width=\linewidth]{figures/contextpose_architecture.pdf}
       ```
       keeping the caption and `\label{fig:arch}`.
     - Note that the optional context skip is left out of the diagram to avoid clutter.
     - Flag the leftover CLIP mentions in the table, the implementation paragraph and the TikZ figure, and the empty "Pose input" paragraph.
   - No tool calls are needed.

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: /Users/henryansah/.claude/projects/-Users-henryansah-workspace-cvpr/67cd9486-2e25-4f58-9bb8-bc406f97875a.jsonl
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.

The messages after this summary are the most recent messages from before compaction, kept verbatim. The summary was written without seeing them, so something it says has not happened yet may already have happened in them.

---


**Claude**

Your run worked. The PDF, SVG and PNG were all written; it only printed the PDF's name, which made it look like that was the only output.

The "'created' timestamp seems very low" lines are harmless. They come from fontTools, which matplotlib uses to embed Arial in the PDF: the date fields in the Arial font file are nearly zero, so fontTools warns and treats them as Unix timestamps. The figure itself isn't affected.

I changed [contextpose_architecture.py](reports/figures/contextpose_architecture.py) so that:
- those warnings are hidden, with a comment saying why;
- the closing line lists all three files written.

I re-ran it. The output is now just `wrote contextpose_architecture.pdf, contextpose_architecture.svg, contextpose_architecture.png`, and the files are the same sizes as before. The same command works as before:

```bash
uv run --no-project --with matplotlib python -I reports/figures/contextpose_architecture.py
```

---


**User**

some discrepancies in image. and make it a little bit more simpler. Too many moving parts are shown

---


**User**

[Image: source: /private/tmp/claude-501/-Users-henryansah-workspace-cvpr/67cd9486-2e25-4f58-9bb8-bc406f97875a/images/1.png]

---


**Claude**

I've redrawn the figure in [contextpose_methodology.tex](reports/contextpose_methodology.tex:64), but I haven't been able to compile it: downloading a LaTeX engine was blocked here, so I haven't seen it rendered. The brace, macro and node-name checks pass, and the measured text widths fit inside the boxes.

**Problems in the old figure**
- **Overlaps and backwards arrows:** a global `text width=1.75cm` wrapped every label onto 3–4 lines. The boxes grew taller than the spacing between them, so the Linear box covered the input boxes and the input → conv arrows pointed up.
- **Labels on borders:** the $\mZ_r$ labels sat on the edges of the Cross-attn. boxes.
- **CLIP:** the context box still listed CLIP.
- **Lookup not linked:** the offset $\vb_{g(\vc)}$ wasn't connected to $\vc$.
- **Shared weights not shown:** three separate conv and attention columns suggested three sets of weights, although they are shared.
- **Face stream:** it wasn't marked optional, and it didn't say where the face features come from.

**What's simpler now**
- **Boxes:** 13 instead of 20.
- **One window column:** the three columns are now one, inside a frame labelled "each window r ∈ {pre, event, post}, shared weights".
- **Context query:** the Linear layer and the region embedding are one "Context query" box.
- **Face stream:** three boxes instead of four, inside a dashed "optional face stream" frame.
- **Natural size:** the `\resizebox` is gone. The figure is about 13.5 cm wide, so its fonts match the text.

If you're editing your own copy, you need two changes. Replace the `\usetikzlibrary` line with this, and add the three styles to your `\tikzset`:

```latex
\usetikzlibrary{positioning,arrows.meta,calc,fit,backgrounds}
% inside \tikzset{...}, after arr/.style:
  winframe/.style={draw=gray!60,rounded corners=4pt,inner sep=2.5mm},
  faceframe/.style={winframe,dashed},
  framelabel/.style={font=\scriptsize,align=center,inner sep=1.5pt},
```

Then replace the whole figure with:

```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}[node distance=5mm]
  % Behaviour stream for one window r; the frame marks modules shared by the three windows
  \node[inp,minimum width=3.8cm] (X) {Behaviour $\mX_r\in\R^{T\times D_x}$\\[-1pt]{\scriptsize pose or EmotionCLIP}};
  \node[op,minimum width=3.8cm,below=of X] (T) {Temporal conv.\\[-1pt]{\scriptsize $\to\mZ_r\in\R^{T\times W}$}};
  \node[op,minimum width=3.8cm,below=of T] (A) {Cross-attention\\[-1pt]{\scriptsize $\to\vs_r\in\R^{W}$}};
  \begin{scope}[on background layer]
    \node[winframe,fit=(X)(T)(A)] (win) {};
  \end{scope}
  \node[framelabel,anchor=south] at (win.north) {each window $r\in\{\text{pre},\text{event},\text{post}\}$\\[-1pt]shared weights};
  \draw[arr] (X) -- (T);
  \draw[arr] (T) -- (A);

  % Fusion and output
  \node[op,minimum width=3.8cm,below=8mm of A] (RT) {Inter-region Transformer\\[-1pt]{\scriptsize mean over windows $\to\vh$}};
  \node[op,minimum width=2.8cm,below=of RT] (H) {MLP head\\[-1pt]{\scriptsize $W\to W/2\to 2$}};
  \node[draw,circle,inner sep=0pt,minimum size=4.5mm,font=\small,below=of H] (plus) {$+$};
  \node[outputnode,minimum width=2.8cm,below=of plus] (Y) {$\vy=[\hat y_{\text{val}},\hat y_{\text{aro}}]$};
  \draw[arr] (A) -- (RT);
  \draw[arr] (RT) -- (H);
  \draw[arr] (H) -- node[right,font=\scriptsize]{$\vy_{\text{beh}}$} (plus);
  \draw[arr] (plus) -- (Y);

  % Context: query for the cross-attention, and key for the fixed lookup offset
  \node[inp,minimum width=3.3cm] (C) at ($(X)+(-4.8,0)$) {Context $\vc$\\[-1pt]{\scriptsize indicators or VLM text}};
  \node[op,minimum width=3.3cm] (Q) at (C |- A) {Context query\\[-1pt]{\scriptsize $\vchat=\mW_c\vc+\vb_c$}\\[-1pt]{\scriptsize $\vq_r=\vchat+\ve_r$}};
  \node[op,minimum width=3.3cm] (Lk) at (C |- plus) {Context lookup\\[-1pt]{\scriptsize offset $\vb_{g(\vc)}$, fixed}};
  \draw[arr] (C) -- (Q);
  \draw[arr] (Q.east) -- node[pos=0.4,above,font=\scriptsize]{$\vq_r$} (Q.east -| A.west);
  \draw[arr] (C.west) -- ++(-3.5mm,0) |- (Lk.west);
  \draw[arr] (Lk.east) -- (plus.west);

  % Optional face stream
  \coordinate (mid) at ($(T)!0.5!(A)$);
  \node[inp,minimum width=3.3cm] (F) at ($(X)+(4.8,0)$) {Face features $\mathbf{x}^{f}_{r,t}$\\[-1pt]{\scriptsize frozen FER ViT}};
  \node[op,minimum width=3.3cm] (FE) at (F |- mid) {Face encoder\\[-1pt]{\scriptsize gated mean $\to\bar{\vu}_r$}};
  \node[op,minimum width=3.3cm] (FH) at (F |- RT) {Face head $\psi$\\[-1pt]{\scriptsize zero-initialised}};
  \begin{scope}[on background layer]
    \node[faceframe,fit=(F)(FE)(FH)] (face) {};
  \end{scope}
  \node[framelabel,anchor=south] at (face.north) {optional face stream};
  \draw[arr] (F) -- (FE);
  \draw[arr] (FE) -- (FH);
  \draw[arr] (FH.south) |- node[pos=0.75,above,font=\scriptsize]{$\bm{\Delta}_{\text{face}}$} (plus.east);
\end{tikzpicture}
\caption{ContextPose. Each window's behaviour sequence (pose features or subject-aware EmotionCLIP embeddings) is encoded by a temporal convolution and summarised by cross-attention, with a query formed from the projected context and a learned region embedding. The framed modules share their weights across the three windows. A Transformer layer relates the three window summaries, which are averaged and decoded to valence and arousal. The output is a residual on a fixed context-lookup offset, and the optional face stream (dashed) adds a zero-initialised correction.}
\label{fig:arch}
\end{figure}
```

The boxes are placed relative to each other's edges, so they shouldn't overlap even if a label comes out wider than measured. One spot is tight: the $\vq_r$ label sits about 3 mm from the frame line. The previous version of the file is saved at `/private/tmp/claude-501/-Users-henryansah--claude-spares-spare-5WONKs/67cd9486-2e25-4f58-9bb8-bc406f97875a/scratchpad/contextpose_methodology.before_tikz_simplify.tex`.

---


**User**

Yes, install it

---


**Claude**

I installed Tectonic and compiled the document, and Figure 1 now looks right: no overlaps, every arrow points the right way, and both frame labels sit clear of the boxes. I've sent you the render.

The first render showed one thing to change. The face encoder sat between two rows, so I moved it level with the cross-attention row, next to the Context query box on the other side. That's the only change from the code I gave you earlier. If you're using your own copy:
- delete the `\coordinate (mid) …` line;
- change `(FE) at (F |- mid)` to `(FE) at (F |- A)`.

The compile gave one warning, and it isn't from the figure: the hyperparameter table (`tab:hparams`) is 35 pt (about 12 mm) too wide and runs into the right margin. The widest row is "Transformer dropout / head dropout $p_h$ / face dropout $p_f$". The table's $d_c$ row also still lists CLIP.

To compile the document yourself from the repository root:

```bash
tectonic -X compile reports/contextpose_methodology.tex
```

This puts the PDF next to the `.tex` file. The first run downloads the LaTeX packages it needs, so it takes longer than later runs.

---


**User**

Leave it

---


**Claude**

OK, I'll leave the table alone. The figure work is done in [contextpose_methodology.tex](reports/contextpose_methodology.tex:64).

---


**User**

@"/Users/henryansah/Downloads/main (5).tex"
I'm writing methodology as a separate tex and adding to this main but image does not show. what's wrong?

---


**Claude**

That error comes from my stand-in style file, not from your files: its version string isn't in the date format LaTeX expects. I'm fixing the stand-in and recompiling both projects.

---


**Claude**

The image is there, but LaTeX moves it near the end of the paper, so it never appears in the methodology. It's also mangled: its symbols are blank and it spills into the other column. I rebuilt your setup and got exactly that: the figure ended up on page 13 of 14. With the fixes below, the same build has no errors and the figure sits across both columns at the top of the methodology page.

### What's wrong

1. **The notation macros are missing.** `\mX`, `\R`, `\vchat` and the rest were defined in my standalone file's preamble, and that block never reached your main file. Every symbol in the figure (and in the equations) comes out empty, and the log shows about 130 "Undefined control sequence" errors.
2. **`\vs` is already taken.** CVPR's `cvpr.sty` defines `\vs` as "vs.", so `\vs_r` gives a "Double subscript" error instead of **s**ᵣ.
3. **The figure is too wide for a column.** It's 13.5 cm wide, and a CVPR column is about 8.3 cm. Inside a plain `figure` it overflows into the next column. It's also too tall for a column with its caption, so LaTeX keeps postponing it, and every figure after it gets postponed too.
4. **Packages load before `cvpr.sty`.**
   - `hyperref` is loaded twice, on lines 51 and 81 of your main file. That's an option-clash error.
   - TikZ loads `xcolor` before `cvpr.sty` asks for `xcolor` with the `dvipsnames` option. That's another option-clash error.
   - The template comment says to load extra packages in `preamble.tex`, and this is why.

### Fix

**1. In `main.tex`,** delete everything from `\usepackage[T1]{fontenc}` down to the closing `}` of `\tikzset`. That's lines 42–62, and it includes the first `hyperref`. `cvpr.sty` already loads `amsmath`, `amssymb`, `booktabs`, `xcolor`, `graphicx` and `caption`, and it sets the Times font.

**2. Add this to the end of `preamble.tex`.** It holds your own TikZ styles, unchanged:

```latex
% --- ContextPose figure and notation (loaded after cvpr.sty, before hyperref) ---
\usepackage{bm}
\usepackage{tabularx}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc,fit,backgrounds}

\tikzset{
  box/.style={draw,rounded corners=2pt,align=center,font=\footnotesize,minimum height=7mm,inner sep=3pt,fill=white},
  inp/.style={box,fill=blue!6},
  op/.style={box,fill=gray!8},
  outputnode/.style={box,fill=green!8},
  arr/.style={-{Stealth[length=2mm]},thick},
  winframe/.style={draw=gray!65,dashed,rounded corners=3pt,inner sep=5pt},
  faceframe/.style={draw=orange!70!black,dashed,rounded corners=3pt,inner sep=5pt},
  framelabel/.style={font=\scriptsize,align=center,fill=white,inner sep=1pt},
}

% Notation
\newcommand{\R}{\mathbb{R}}
\newcommand{\regions}{\mathcal{R}}
\newcommand{\vc}{\mathbf{c}}
\newcommand{\vchat}{\hat{\mathbf{c}}}
\newcommand{\ve}{\mathbf{e}}
\newcommand{\vq}{\mathbf{q}}
\newcommand{\vsum}{\mathbf{s}} % \vs is taken by cvpr.sty ("vs.")
\newcommand{\vh}{\mathbf{h}}
\newcommand{\vy}{\hat{\mathbf{y}}}
\newcommand{\vb}{\mathbf{b}}
\newcommand{\vu}{\mathbf{u}}
\newcommand{\mZ}{\mathbf{Z}}
\newcommand{\mX}{\mathbf{X}}
\newcommand{\mS}{\mathbf{S}}
\newcommand{\mW}{\mathbf{W}}
\newcommand{\GELU}{\operatorname{GELU}}
\newcommand{\ReLU}{\operatorname{ReLU}}
\newcommand{\LN}{\operatorname{LN}}
\newcommand{\Drop}{\operatorname{Drop}}
\newcommand{\softmax}{\operatorname{softmax}}
```

**3. In `methodology_h.tex`,** make two changes:
- Change the figure to `\begin{figure*}[t]` … `\end{figure*}`. The text block is 17.5 cm wide, so the figure fits without scaling. A `figure*` always goes at the top of a page, at the earliest the page after where it appears in the source, so keep it near the start of the section.
- Replace every `\vs_` with `\vsum_`. My version of the file has 7 of them.

### How I tested it
I couldn't use the real `cvpr.sty`, so I built against a stand-in that loads the same packages, sets the same page width, and defines `\vs` the same way. I also used my copy of the methodology, not your `methodology_h.tex`. If Overleaf still reports an error after these changes, its log will name the line.

The methodology was written for one column, so a few other things don't fit CVPR's columns:
- three equations run 2–3.5 cm too wide: the attention equation, the face projection and the face-correction cases;
- the hyperparameter table is about 8.6 cm too wide.

---


**User**

Yes, fit them

---


**Claude**

Equations 8, 9 and 12 and the table now fit their columns. The attention equation is on page 7, so I'm checking that page too.

---


**Claude**

The equations and the table now fit CVPR's columns. In the two-column test build, the methodology compiles with no errors and nothing runs past its column edge, and I checked the rendered pages by eye. I made the same changes in [contextpose_methodology.tex](reports/contextpose_methodology.tex), which still compiles cleanly in one column.

To update your `methodology_h.tex`, replace these blocks. They already use `\vsum`.

**1. Attention equation**: the two formulas now sit on two lines, aligned at "=".
```latex
\begin{equation}
  \begin{aligned}
    \bm{\alpha}_r &= \softmax\!\left(\frac{(\vq_r\mW^{Q})(\mZ_r\mW^{K})^{\top}}{\sqrt{d_h}}\right)\in\R^{T},\\
    \vsum_r &= \sum_{t=1}^{T}\alpha_{r,t}\,\mathbf{z}_{r,t}\mW^{V}.
  \end{aligned}
  \label{eq:attn}
\end{equation}
```

**2. Face projection**: the same two-line treatment.
```latex
\begin{equation}
  \begin{aligned}
    \vu_{r,t} &= \GELU\big(\mW_f\,\LN(\mathbf{x}^f_{r,t})+\vb_f\big)\in\R^{32},\\
    \bar{\vu}_r &= \frac{\sum_{t} m_{r,t}\,\vu_{r,t}}{\max\big(1,\sum_{t} m_{r,t}\big)}.
  \end{aligned}
\end{equation}
```

**3. Face correction (`eq:face`)**: only the first case changes. The words "if the clip has at least one valid face step" become the condition $\sum_{r,t} m_{r,t}>0$, which means the same thing. The paragraph after the equation still reads correctly.
```latex
    \psi\big([\bar{\vu}_{\text{pre}};\bar{\vu}_{\text{event}};\bar{\vu}_{\text{post}}]\big), & \text{if } \sum_{r,t} m_{r,t}>0,\\
```

**4. Loss equation**: in the sum's subscript, delete the `\,` between `\text{val},` and `\text{aro}`. That equation was 0.6 pt too wide.

**5. Hyperparameter table**: it now has two columns, and the old "Component" column is gone. The longest rows are shortened or split, and "(indicator context)" moved into the caption. All values are unchanged. It uses `[t]` instead of `[h]`, which suits a two-column paper better.
```latex
\begin{table}[t]
\centering
\caption{Hyper-parameters of ContextPose. Parameter counts use indicator context.}
\label{tab:hparams}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lr@{}}
\toprule
Hyper-parameter & Value \\
\midrule
Steps per window $T$ / joints $J$ / features $F$ & 32 / 11 / 4 \\
Behaviour input $D_x$ (pose / EmotionCLIP) & 44 / 513 \\
Context $d_c$ (indicators / CLIP / VLM) & 4 / 768 / 384 \\
Conv.\ layers / kernel / activation & 2 / 3 / GELU \\
Model width $W$ & 64 \\
Attention heads $H$ / head width $d_h$ & 4 / 16 \\
Region Transformer layers / heads / FFN & 1 / 4 / $2W$ \\
Face input $D$ / encoding width & 768 / 32 \\
Prediction head hidden width & $W/2$ \\
Dropout: Transformer, head $p_h$, face $p_f$ & 0.3 \\
\midrule
Optimiser & AdamW \\
Learning rate / weight decay & $3{\times}10^{-4}$ / 0.05 \\
Batch size / max.\ epochs / patience & 16 / 60 / 10 \\
Ensemble size (seeds) & 5 \\
\midrule
Parameters, pose (no face / face) & 73.6k / 102.9k \\
Parameters, EmotionCLIP (no face / face) & 163.7k / 193.0k \\
\bottomrule
\end{tabular}
\end{table}
```

The context row still lists CLIP, as your table did. Drop "CLIP / 768" from it if CLIP context is out of the paper.

---


**User**

what should I do now to apply the fixes

---


**Claude**

You need to edit three files in your Overleaf project. All the code to paste is in my previous two messages.

### 1. `main.tex`
Delete lines 42–62: everything from `\usepackage[T1]{fontenc}` down to the `}` that closes `\tikzset{`. That removes the first `hyperref`. Keep the one on line 81.

### 2. `preamble.tex`
Paste the "ContextPose figure and notation" block at the end. It's the one starting `% --- ContextPose figure and notation`, with the packages, your `\tikzset` and the `\newcommand` list. If Overleaf later reports "Command … already defined", that macro already exists in `preamble.tex`, so delete one of the two definitions.

### 3. `methodology_h.tex`
1. The file should hold only the section text. If it still has `\documentclass`, `\usepackage`, `\newcommand`, `\begin{document}`, `\maketitle` or `\end{document}`, delete those lines.
2. For the architecture figure only, change `\begin{figure}[t]` to `\begin{figure*}[t]` and `\end{figure}` to `\end{figure*}`.
3. Open Find & Replace (Ctrl/Cmd+F) and replace `\vs_` with `\vsum_`. Use Replace All; it won't touch `\vspace`.
4. Paste in the five fixes from my last message:
   - the attention equation (`eq:attn`);
   - the face projection equation (the one defining `\vu_{r,t}` and `\bar{\vu}_r`);
   - the first line of the `eq:face` cases;
   - the loss equation, deleting the `\,` between `\text{val},` and `\text{aro}`;
   - the hyperparameter table, which replaces the whole old `table` block.

### 4. Recompile from scratch
The `hyperref` setup changed, so clear the old build files first. In Overleaf, click the Logs icon next to Recompile, choose "Clear cached files", then recompile. Then check:
- the log has no errors, and in particular no "Undefined control sequence" or "Option clash";
- the architecture figure spans both columns at the top of a page near the start of the methodology, with all its symbols showing;
- the equations and the table stay inside their column.

If you'd rather not edit by hand, put your current `methodology_h.tex` and `preamble.tex` in the repo or paste them here. I'll apply everything, test it in the CVPR layout, and give you the finished files.

---


**User**

@"/Users/henryansah/Downloads/methodology_h.tex" @"/Users/henryansah/Downloads/preamble.tex"

---


**Claude**

I've applied every fix and compiled your three files against the CVPR stand-in. The build has no errors and nothing runs past a column edge. Last step is handing them over.

---


**Claude**

Your three files are fixed and ready to upload. Built with the CVPR stand-in, they compile with no errors and nothing runs past a column edge. The architecture figure spans both columns at the top of the methodology page, and the table fits one column. Your files in Downloads are unchanged.

### Files to upload
In Overleaf, replace these with the copies in `reports/overleaf/`:
- [main.tex](reports/overleaf/main.tex)
- [preamble.tex](reports/overleaf/preamble.tex)
- [methodology_h.tex](reports/overleaf/methodology_h.tex)

Then open the Logs menu, choose "Clear cached files", and recompile.

### What changed
- **`main.tex`:**
  - Removed lines 42–62, which loaded packages before `cvpr.sty`, including the duplicate `hyperref`.
  - Commented out the leftover `\label{sec:method}` after `% \section{Method}`. `methodology_h.tex` defines the same label.
  - Fixed `Fig.~\ref{fig:pixels_vs_normalized_pose}`, which pointed to a label that doesn't exist and showed "Fig. ??". It now uses `fig:normalization`.
- **`preamble.tex`:** your file plus the TikZ packages, your figure styles and the notation macros, with `\vsum` instead of `\vs`.
- **`methodology_h.tex`:**
  - The figure is now `figure*`.
  - `\vs_` is now `\vsum_` in all 6 places, including inside the figure.
  - Your table was commented out, so `Table~\ref{tab:hparams}` would have shown "??". I put back the column-width version.
  - Your equations already had the earlier fixes and didn't need changes.

### How I tested it
I used a stand-in for `cvpr.sty`, and the other sections, the tables you `\input`, and the bibliography were left empty. If Overleaf's log shows an error after you upload, it names the file and line.

### Content issues I left alone
- **Two representations, not three:** the Context encoder section says "three alternative representations" but now describes only two, indicators and VLM.
- **CLIP is still mentioned in three places:**
  - the implementation paragraph ("VLM or CLIP features … 98.0k and 122.5k");
  - the epochs sentence ("60 epochs for indicator and CLIP contexts");
  - the table's $d_c$ row.
- **Citation:** "(Zhang et al., 2023)" for EmotionCLIP is plain text; it should be a `\cite{…}`.
- **Section numbering:** `\section{Method}` in `main.tex` is commented out, so the behaviour-recognition subsections that follow (Problem Formulation, Pose Extraction, …) are numbered 3.11, 3.12 and so on under "Context Pose". You'll probably want that `\section` back, under a name that fits.

---


**User**

show me what to change in main. don't want to paste everything from here

---


**Claude**

`main.tex` needs three edits, and all of them are deletions or one-word changes, so there's nothing to paste. The line numbers are from your current file.

**1. Delete lines 42–62.** That's the block from `\usepackage[T1]{fontenc}` down to the `}` that closes `\tikzset{`, starting:
```latex
\usepackage[T1]{fontenc}
\usepackage{lmodern}
...
\usepackage[colorlinks=true,linkcolor=blue!50!black,...]{hyperref}

\tikzset{
  ...
}
```
After this, `\documentclass[10pt,twocolumn,letterpaper]{article}` is followed straight by the `%%%%%%%%% PAPER TYPE` comment. The second `hyperref` (line 81, with `pagebackref`) stays. The packages and styles you deleted move to `preamble.tex`.

**2. Line 212.** Comment out the label under `% \section{Method}`:
```latex
% \label{sec:method}
```

**3. Line 238.** In the "Normalization and per-frame input" paragraph, change
`\ref{fig:pixels_vs_normalized_pose}` → `\ref{fig:normalization}`

Edits 1 and 2 only work if `preamble.tex` and `methodology_h.tex` are updated too. Otherwise the figure loses its styles and macros.

**`preamble.tex`:** add the block at the end of [reports/overleaf/preamble.tex](reports/overleaf/preamble.tex), from `%% ContextPose figure and notation` down.

**`methodology_h.tex`:**
- change `figure` to `figure*` (both the begin and the end);
- replace `\vs_` with `\vsum_`;
- un-comment the table, or use the narrower version in [reports/overleaf/methodology_h.tex](reports/overleaf/methodology_h.tex).

Then clear the cached files in Overleaf and recompile.

---


**User**

what is the reason for step 3?

---


**Claude**

Step 3 fixes a reference to a label that doesn't exist. It isn't related to the ContextPose figure: it was already broken in your `main.tex`, and my test compile caught it.

The normalisation figure is labelled `\label{fig:normalization}` (around line 234), but the text that refers to it uses `\ref{fig:pixels_vs_normalized_pose}`, which is the image's file name, not its label. LaTeX can't find that label, so the paper prints "(Fig. ??)" and the log shows "Reference `fig:pixels_vs_normalized_pose` undefined".

Pointing the `\ref` at `fig:normalization` makes it print the right figure number. You could instead rename the label to match the `\ref`; either works as long as the two agree.

---


**User**

create a regression metrics table with metrics such as MAE, RMSE etc like we have in table 2 for the different combinations we have in table 1

---


**Claude**

Now a test with synthetic reports, so the script is checked before you run it on the real ones.

---


**Claude**

Tests and lint pass. Last check: rendering the generated LaTeX from synthetic reports, to confirm the table compiles and reads like Table 2.

---


**Claude**

The table renders correctly; the numbers in it are synthetic. One small fix: values that round to zero print as "−0.000".

---


**Claude**

I've written a script that builds the table, but I couldn't fill in the numbers here. The Table 1 runs and their `cv_report.json` files are on your GPU machine (`checkpoints/streams/...`), and this machine has neither the data nor the run outputs. You'll need to run the script there.

**What it does.** [regression_table.py](src/asd_contextpose/regression_table.py) reads the saved Table 1 runs, so nothing is retrained. Each run's report already stores every out-of-fold prediction. For each stream combination the script:
- averages the 5 seeds and works out which streams (P/A/V/F) the run used from its training arguments, printing that mapping so you can check it;
- writes a LaTeX table in Table 2's layout, with Table 1's checkmark columns and rows in Table 1's order.

The columns are:
- **Mean MAE, mean RMSE, mAP**;
- **V MAE / RMSE / CCC** and **A MAE / RMSE / CCC**;
- two reference rows: always predicting the training folds' mean, and the event-annotation lookup.

**One difference from Table 2.** Table 2 scores the consensus label on a [−1, 1] scale. The Table 1 models were trained on vote shares, so they predict the probability that an annotator votes positive valence or high arousal. The script compares that predicted share with the observed vote share, on a [0, 1] scale. Converting it to the consensus scale would be wrong for valence: "not positive" mixes neutral and negative, and the model can't tell them apart. So the new table's MAE and RMSE can't be compared number-for-number with Table 2's, and the caption says so.

On the GPU machine, after copying over the two new files (they aren't committed yet), run:

```bash
uv run python -m asd_contextpose.regression_table checkpoints/streams --output reports/regression_table.tex --json reports/regression_table.json
```

Before using the table:
- Check the printed list. There should be one line per Table 1 row, each with `seeds=5`.
- If the VLM runs sit in another folder, such as `checkpoints/v3vlm`, add that folder to the command.
- If two settings share the same streams, for example the 150-epoch VLM runs, the script warns. Then pass only the folders that Table 1 used.

**How I tested it.** The tests in [test_regression_table.py](tests/test_regression_table.py) pass (13 of them): stream detection, average precision, perfect predictions, the reference rows, and a full run on synthetic reports. I also compiled a sample table from synthetic reports and checked it by eye; it looks right, but none of its numbers are real.

If you want numbers that can be compared directly with Table 2, the alternative is to retrain all the combinations with the regression objective. That means 16 new cross-validation runs on the GPU machine, each with 5 seeds.

---
