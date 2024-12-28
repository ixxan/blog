---
title:  "Speech Technology for Low-Resource Languages: Advancements and Resources for Uyghur"
author: "Irpan Abdurahman"
mathjax: false
layout: post
categories: media
---

<img src="./../images/speech.jpg" alt="speech" style="max-width: 100%; height: auto;">

Language is a fundamental part of human culture, shaping our identities, traditions, and ways of thinking. With over 7,000 languages spoken globally, it's concerning that nearly half are endangered <a href="https://www.ethnologue.com" style="text-decoration: none;" target="_blank" title="Visit Ethnologue for more info">[1]</a>.



In the past decade, speech technologies integrated into smartphones and other personal devices have become essential to our daily lives. However, these systems only support a handful of languages. As AI-driven speech systems continue to evolve, the reliance on these limited language systems could deepen the digital divide, further marginalizing underrepresented languages. This is especially concerning for younger generations in minority communities. While immersed in technology from an early age, children may primarily interact with dominant languages, even as families work to preserve their native tongues.

Thus, through this blog, I would like to:
1. Offer a tutorial on the fundamentals of speech technology for those interested
2. Explore recent advancements in speech technology research for low-resource languages
3. Share valuable resources and highlight specific implementations in my native language, Uyghur

By addressing these topics, I hope to inspire further efforts in preserving linguistic diversity and advancing speech technologies that can support a wider range of languages.

# Table of Contents
1. [Speech Technology Fundamentals](#speech-technology-fundamentals)
    * [Phonetics Basics](#phonetics-basics)
    * [Acoustics Basics](#acoustics-basics)
    * [Speech Applications](#speech-applications)
    * [Notable Architectures](#notable-architectures)
2. [Low-Resource Language Speech Research](#low-resource-language-speech-research)
3. [Uyghur Speech Technology](#uyghur-speech-technology)
4. [References](#references)

# Speech Technology Fundamentals
<a id="speech-technology-fundamentals"></a>
Speech technology has evolved significantly in the past decade, enabling devices to understand and generate human speech. To fully appreciate these advancements, it’s important to understand the fundamentals of speech systems.

## Phonetics Basics
<a id="phonetics-basics"></a>
Phonetics studies the sounds in human speech, vital for converting between written text and spoken language.

A **grapheme** is the smallest unit of a writing system that represents a sound or a meaningful unit of language. For example, in English, the letter “b” is a grapheme representing the /b/ sound. 

A **phoneme** is the smallest unit of sound in a language that can distinguish one word from another. Phonemes are auditory representations, not written forms like graphemes, and can vary depending on the speaker’s accent, language, and context. For example, in English, /p/ and /b/ are distinct phonemes because changing them alters the meaning of a word, such as changing “pat” (/p/) to “bat” (/b/). The **International Phonetic Alphabet (IPA)** is a standardized system of symbols that represents phonemes across languages.

With graphemes and phonemes, we can create **grapheme-to-phoneme (G2P) mapping**, which converts written text into its corresponding phonetic representation, crucial for speech recognition and synthesis. 

Python libraries like [epitran](https://github.com/dmort27/epitran) can be used to transliterate text to IPA phoneme for many languages, while libraries like [g2p-en](https://pypi.org/project/g2p-en/) and [g2p-seq2seq](https://github.com/cmusphinx/g2p-seq2seq) has also been developed for G2P mapping in multiple dominate languages. 

<img src="./../images/engIPA.jpg" alt="IPA" style="max-width: 100%; height: auto;">

*The 44 phonemes of Standard English based on [International Phonetic Alphabet](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet).*

## Acoustics Basics
<a id="acoustics-basics"></a>

Acoustics studies how waves travel through media like air or water, crucial for understanding human speech and building audio processing systems.
<img src="./../images/vocal.png" alt="Vocal" style="float: right; max-width: 30%; margin-left: 20px; margin-top: 20px; margin-bottom: 20px; height: auto;">

Speech production begins with our vocal folds creating a basic sound when air passes through them. However, before the sound exits the mouth, it travels through the vocal tract (throat, mouth, and nose) which shapes and modifies it. These modifications produce **formants**, which are resonant frequencies essential for distinguishing sounds in a language, especially vowels <a href="http://www.voicescienceworks.org/harmonics-vs-formants.html" style="text-decoration: none;" target="_blank" title="Visit Voice Science Works for more info">[2]</a>.

To analyze these acoustic properties, we use tools like the spectrogram. A **spectrogram** is a visual representation of the spectrum of frequencies in a sound signal over time. It shows how the frequency content of an audio signal changes, allowing us to visualize speech as a sequence of sounds. The **Mel spectrogram** is a variation that uses the Mel scale, which aligns more closely with human hearing. The Mel scale compresses high-frequency components, where our hearing is less sensitive, and expands low-frequency components, where we are more sensitive. Mel spectrograms are widely used as input features for speech models <a href="https://huggingface.co/learn/audio-course/en/chapter1/audio_data" style="text-decoration: none;" target="_blank" title="Check out this HuggingFace course to learn more">[3]</a>.

The effectiveness of spectrograms depends on the **sampling rate**, the number of times per second an audio signal is sampled during digitization. A higher sampling rate provides more detail, capturing a clearer representation of the sound. However, higher sampling rates also require more computational power and storage. For speech processing, **16 kHz** is commonly used, as it captures frequencies up to 8 kHz, covering most of the important information in human speech. For music, a higher sampling rate like **44.1 kHz** is standard, capturing frequencies up to 22.05 kHz, the upper limit of human hearing.

Libraries like [ibrosa](https://librosa.org/doc/latest/index.html) and [pydub](https://github.com/jiaaro/pydub) simplify tasks such as generating spectrograms, adjusting sampling rates, and feature extraction, aiding robust speech processing.

<img src="./../images/melSpec.png" alt="melSpec" style="clear: both; max-width: 100%; height: auto;">

*Comparison of Mel spectrograms for an audio clip saying "Hello World" at 16 kHz and 44.1 kHz sampling rates. Higher sampling rates capture more detailed information, as seen in the richer frequency representation at 44.1 kHz.*

## Speech Applications
<a id="speech-applications"></a>
Speech technology encompasses a broad range of applications that facilitate various tasks. Among these, Automatic Speech Recognition and Speech Synthesis are two of the most influential and widely used.

**Automatic Speech Recognition (ASR)**, also known as **Speech-to-Text (SST)**, converts human speech into written text. ASR at the foundation of numerous technologies, such as virtual assistants, transcription services, real-time translation systems, and accessibility tools. To build a reliable ASR system, large datasets of transcribed speech, covering diverse speech patterns, accents, and languages, are essential. Its performance is typically evaluated using metrics like **Word Error Rate (WER)** or **Character Error Rate (CER)** on datasets such as [FLEURS](https://huggingface.co/datasets/google/fleurs) (a widely recognized benchmark for multilingual ASR, covering 102 languages) <a href="https://huggingface.co/learn/audio-course/en/chapter1/audio_data" style="text-decoration: none;" target="_blank" title="Check out this HuggingFace course to learn more">[4]</a>.

**Speech Synthesis**, also known as **Text-to-Speech (TTS)**, converts written text into natural-sounding speech.  It is used in virtual assistants, audiobooks, language learning platforms, and tools for visually impaired users. High-quality TTS models typically need labeled audio data from a single speaker to produce realistic voices. Evaluation metrics include subjective measures like **Mean Opinion Score (MOS)** and objective metrics like **Word Error Rate (WER)** or **Character Error Rate (CER)** to assess the naturalness and accuracy of synthesized speech.

Some other common speech technology tasks include:
* **Audio Classification**: Categorizing audio clips into distinct classes.
* **Language Identification**: Identifying the language spoken in an audio clip.
* **Speaker Diarization**: Identifying the speaker at any given moment in an audio clip.
* **Keyword Detection**: Detecting specific words or phrases in continuous speech, often used for wake-word detection in voice assistants.

## Notable Architectures
<a id="notable-architectures"></a>
There have been various architectures proposed for various speech tasks throughout the years. Here I would discuss 2 popular types. 

### Traditional Approach: Hidden Markov Models (HHM)
Hidden Markov Models (HMMs) were the foundation of many early ASR and TTS systems before the rise of deep learning-based models. They model speech as sequences of phonemes with probabilities for transitions and observations. 

To better understand it, let’s try to recognize the word “hello” from a speech signal using HMMs. Here’s our set up:
* **Hidden States (Phonetic Features)**: represent the phonemes. (e.g., /h/, /e/, /l/, /o/ for “hello”).
* **Observations (Acoustic Features)**: the actual audio features that we can measure, such as frequency or amplitude. (e.g., we observe the distinct frequency for /e/).
* **Transition Probabilities**: Likelihood of moving between states (e.g., 0.7 for /h/ → /e/).
* **Emission Probabilities**: Likelihood of audio features given a current state. (e.g., 0.8 for frequency around 1800 Hz given /e/).

Now to walk through the recognizing process: 
1. We start with an initial probability for the first state. For example, we might say the probability of starting with the phoneme /h/ is 1 (since we know the word starts with “h”).
2. The model then transitions from state to state based on the transition probabilities. For example, after hearing /h/, it is likely to transition to /e/.
3. For each state (phoneme), we generate an observation based on the emission probabilities. For example, in state /h/, the observation might be a burst of air.
4. Finally, after processing the sequence of phonemes (states) and their corresponding observations, the model uses the transition and emission probabilities to determine the most likely sequence of phonemes that led to the observed acoustic features, which may led to /h/, /e/, /l/, /l/, /o/, or “hello.”


### Modern Approach: End-to-End (E2E)
With the rise of deep learning methods in ML since the 2010s, researchers have achieved significant success with deep learning-based speech models, particularly End-to-End (E2E) architectures. E2E architectures eliminate the need for hand-crafted features and separate language and acoustic models by integrating both into a single neural network. Two common E2E structures are based on Connectionist Temporal Classification (CTC) loss functions and Sequence-to-Sequence (Seq2Seq) modeling. Below is how the word “hello” can be recognized using each approach:

**CTC (Connectionist Temporal Classification)**: Encoder-only structure <a href="https://distill.pub/2017/ctc/" style="text-decoration: none;" target="_blank" title="Check out this tutorial to learn more">[5]</a>
1. **Encoder**: The model processes the audio, breaking it into small frames (e.g., 20 milliseconds) and creates a hidden state for each frame.
2. **Vocabulary**: For each hidden state, the model predicts a label, such as a character or phoneme. (e.g., “HHEEE”, is predicted from the first 5 frames)
3. **Blank Token**: Since the model doesn’t know the exact timing of words, it uses a blank token (_) to skip over uncertain regions of the audio.
4. **Post-processing**: After the model makes predictions, repeated labels and blanks are cleaned up (e.g., “HHEEE_ELLL_OO” becomes “HELLO”).

Popular CTC-based ASR Models: [Wav2Vec](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2), [HuBERT](https://huggingface.co/docs/transformers/en/model_doc/hubert), [M-CTC-T](https://huggingface.co/docs/transformers/en/model_doc/mctct).

<video src="./../images/ctc.mov" autoplay loop muted style="max-width: 100%; height: auto;"></video>

*Demo of how CTC algorithm recognizes "hello" from input audio by [Awni Hannun](https://distill.pub/2017/ctc/)*

**Seq2Seq (Sequence-to-Sequence)**: Encoder-Decoder structure <a href="https://huggingface.co/learn/audio-course/en/chapter3/seq2seq" style="text-decoration: none;" target="_blank" title="Check out this tutorial to learn more">[6]</a>
1. **Encoder**: The model listens to an audio clip (e.g., “hello”) and transforms it into a sequence of features (e.g., spectrogram). The encoder processes these features to capture key information about the speech.
2. **Decoder**: The decoder uses the encoded information to generate the output sequence. It starts with a “start” token and predicts the next word or character (e.g., “H” for “hello”), adding each prediction to the growing output.
3. **Cross-Attention**: The decoder uses the encoder’s hidden states to focus on relevant parts of the input when predicting subsequent tokens.
4. **Autoregressive Generation**: The decoder generates one token at a time, using the previous token to predict the next. For example, after predicting “H,” it uses that to predict “E,” and so on, until it finishes with ‘HELLO’

Popular Seq2Seq-based ASR models: [Whisper](https://openai.com/index/whisper/)

Popular Seq2Seq-based TTS models: [SpeechT5](https://github.com/microsoft/SpeechT5), [Tacotron](https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/)

<img src="./../images/seq2seq.png" alt="seq2seq" style="clear: both; max-width: 100%; height: auto;">

*The encoder-decoder structure shown in [OpenAI Whisper blog](https://openai.com/index/whisper/)*

## Low-Resource Language Speech Research
<a id="low-resource-language-speech-research"></a>

**Low-resource languages** are languages with limited linguistic data resources, such as annotated text corpora or speech data, which are essential for building robust speech and language models. In recent years, efforts by various research groups and organizations have led to some significant progress in data creation and model development for low-resource languages.

### Common Voice (2017)
Common Voice is a crowdsourcing project initiated by Mozilla to create a diverse **open-source dataset** of human voice recordings. It encourages volunteers to contribute their voices in various languages, including many low-resource ones. Anyone can record themselves reading texts or validate recordings submitted by others in available languages. It also allows individuals to request new languages and assist in making them available. This community-driven effort has been crucial in expanding and diversifying the dataset. 

The project is continually expanding, with new releases every three months. The latest release, **Common Voice Corpus 20.0**, includes over **22,000 hours** of validated recordings across **130+ languages** <a href="https://commonvoice.mozilla.org" style="text-decoration: none;" target="_blank" title="Visit Mozilla Common Voice for more info">[7]</a>.

### CMU Wilderness (2019) 

The CMU Wilderness Multilingual Speech dataset, developed by Carnegie Mellon University, provides aligned sentences and audio for over **700 languages**, focusing primarily on low-resource languages. The dataset was created by extracting audio from **New Testament** recordings and includes on average **20 hours of data per language**. Importantly, the data creation algorithms were **open-sourced**, enabling others to recreate the same datasets <a href="https://github.com/festvox/datasets-CMU_Wilderness" style="text-decoration: none;" target="_blank" title="Visit CMU Wilderness Github Repo for more info">[8]</a>. This extensive dataset has enabled researchers to investigate the effectiveness of various pre-trained speech models on low-resource languages. 

### ASR2K (2022)

The ASR2K project, also from Carnegie Mellon University, aimed to improve **automatic speech recognition** for **1,909 languages** by leveraging transfer learning and multilingual models. Their approach involves mapping the output of multilingual models trained on high-resource languages to the appropriate phonemes for the target language. ASR2K represents the first attempt to build a speech recognition pipeline for thousands of languages without requiring audio <a href="https://arxiv.org/abs/2209.02842" style="text-decoration: none;" target="_blank" title="Check out the ASR2K paper to learn more">[9]</a>.

### Meta MMS (2023)

In 2023, Meta introduced its Massively Multilingual Speech (MMS) project, which extended the capabilities of **automatic speech recognition** and **text-to-speech** systems to over **1,100 languages** and **language identification** to over **4,000 languages**.

The project utilized **Wav2Vec 2.0 models (CTC-based)** trained on datasets created using self-supervised learning and data alignment techniques. These datasets include a labeled dataset (MMS-lab) covering 1,107 languages and an unlabeled dataset (MMS-unlab) covering 3,809 languages, both created from recordings of New Testaments similar to the CMU Wilderness dataset. Meta also **open-sourced** their data alignment algorithms, enabling others to replicate and build upon their work. <a href="https://github.com/facebookresearch/fairseq/tree/main/examples/mms" style="text-decoration: none;" target="_blank" title="Visit MMS Github Repo for more info">[10]</a>.

Meta-MMS models outperformed some existing state-of-the-art models, including OpenAI's Whisper, with half the word error rate while covering 11 times more languages <a href="https://arxiv.org/abs/2305.13516" style="text-decoration: none;" target="_blank" title="Check out the MMS paper to learn more">[11]</a>. These models are not only some of the most comprehensive and high-performing in the field but are also open-sourced, allowing researchers and developers to fine-tune them for specific applications, paving the way for more inclusive and accessible speech technologies worldwide. Check out examples of how you may fine-tune some of these models for specific low-resource langague in the next section.


## Uyghur Speech Technology
<a id="uyghur-speech-technology"></a>
ABC

## Reference
<a id="references"></a>
You can also embed a lot of stuff, for example from YouTube, using the `embed.html` include.