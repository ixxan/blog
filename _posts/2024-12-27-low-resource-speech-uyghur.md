---
title:  "Speech Technology for Low-Resource Languages: Advancements and Resources for Uyghur"
author: "Irpan Abdurahman"
mathjax: false
layout: post
categories: media
---

<img src="./../images/speech.jpg" alt="speech" style="max-width: 100%; height: auto;">

Language is a fundamental part of human culture, shaping our identities, traditions, and ways of thinking. With over 7,000 languages spoken globally, it's concerning that nearly half are endangered<a href="https://www.ethnologue.com" style="text-decoration: none;" target="_blank" title="Visit Ethnologue for more info"><sup>1</sup></a>.



In the past decade, speech technologies integrated into smartphones and other personal devices have become essential to our daily lives. However, these systems only support a handful of languages. As AI-driven speech systems continue to evolve, the reliance on these limited language systems could deepen the digital divide, further marginalizing underrepresented languages. This is especially concerning for younger generations in minority communities. While immersed in technology from an early age, children may primarily interact with dominant languages, even as families work to preserve their native tongues.

Thus, through this blog, I would like to:
1. Offer a tutorial on the fundamentals of speech technology for those interested
2. Explore recent advancements in speech technology research for low-resource languages
3. Share valuable resources and highlight specific implementations in my native language, Uyghur

By addressing these topics, I hope to inspire further efforts in preserving linguistic diversity and advancing speech technologies that can support a wider range of languages.

# Table of Contents
1. [Speech Technology Fundamentals](#speech-technology-fundamentals)
    1. [Phonetics Basics](#phonetics-basics)
    2. [Acoustics Basics](#acoustics-basics)
    3. [Speech Applications](#speech-applications)
    4. [Notable Architectures](#notable-architectures)
2. [Low-Resource Language Speech Research](#low-resource-language-speech-research) 
3. [Uyghur Speech Technology](#uyghur-speech-technology)
    1. [Brief History of Uyghur Language and Speech Technology](#brief-history-of-uyghur-language-and-speech-technology)
    2. [Open Source Uyghur Speech Datasets](#open-source-uyghur-speech-datasets)
    3. [Open Source Uyghur Speech Models](#open-source-uyghur-speech-models)
    4. [Additional Tools](#additional-tools)
4. [References](#references)

# 1. Speech Technology Fundamentals
<a id="speech-technology-fundamentals"></a>
Speech technology has evolved significantly in the past decade, enabling devices to understand and generate human speech. To fully appreciate these advancements, it’s important to understand the fundamentals of speech systems.

## 1.1 Phonetics Basics
<a id="phonetics-basics"></a>
Phonetics studies the sounds in human speech, vital for converting between written text and spoken language.

A **grapheme** is the smallest unit of a writing system that represents a sound or a meaningful unit of language. For example, in English, the letter “b” is a grapheme representing the /b/ sound. 

A **phoneme** is the smallest unit of sound in a language that can distinguish one word from another. Phonemes are auditory representations, not written forms like graphemes, and can vary depending on the speaker’s accent, language, and context. For example, in English, /p/ and /b/ are distinct phonemes because changing them alters the meaning of a word, such as changing “pat” (/p/) to “bat” (/b/). The **International Phonetic Alphabet (IPA)** is a standardized system of symbols that represents phonemes across languages.

With graphemes and phonemes, we can create **grapheme-to-phoneme (G2P) mapping**, which converts written text into its corresponding phonetic representation, crucial for speech recognition and synthesis. 

Python libraries like [epitran](https://github.com/dmort27/epitran) can be used to transliterate text to IPA phoneme for many languages, while libraries like [g2p-en](https://pypi.org/project/g2p-en/) and [g2p-seq2seq](https://github.com/cmusphinx/g2p-seq2seq) have also been developed for G2P mapping in multiple dominant languages. 

<img src="./../images/engIPA.jpg" alt="IPA" style="max-width: 100%; height: auto;">

*The 44 phonemes of Standard English based on [International Phonetic Alphabet](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet).*

## 1.2 Acoustics Basics
<a id="acoustics-basics"></a>

Acoustics studies how waves travel through media like air or water, crucial for understanding human speech and building audio processing systems.
<img src="./../images/vocal.png" alt="Vocal" style="float: right; max-width: 30%; margin-left: 20px; margin-top: 20px; margin-bottom: 20px; height: auto;">

Speech production begins with our vocal folds creating a basic sound when air passes through them. However, before the sound exits the mouth, it travels through the vocal tract (throat, mouth, and nose) which shapes and modifies it. These modifications produce **formants**, which are resonant frequencies essential for distinguishing sounds in a language, especially vowels<a href="http://www.voicescienceworks.org/harmonics-vs-formants.html" style="text-decoration: none;" target="_blank" title="Visit Voice Science Works for more info"><sup>2</sup></a>.

To analyze these acoustic properties, we use tools like the spectrogram. A **spectrogram** is a visual representation of the spectrum of frequencies in a sound signal over time. It shows how the frequency content of an audio signal changes, allowing us to visualize speech as a sequence of sounds. The **Mel spectrogram** is a variation that uses the Mel scale, which aligns more closely with human hearing. The Mel scale compresses high-frequency components, where our hearing is less sensitive, and expands low-frequency components, where we are more sensitive. Mel spectrograms are widely used as input features for speech models<a href="https://huggingface.co/learn/audio-course/en/chapter1/audio_data" style="text-decoration: none;" target="_blank" title="Check out this HuggingFace course to learn more"><sup>3</sup></a>.

The effectiveness of spectrograms depends on the **sampling rate**, the number of times per second an audio signal is sampled during digitization. A higher sampling rate provides more detail, capturing a clearer representation of the sound. However, higher sampling rates also require more computational power and storage. For speech processing, **16 kHz** is commonly used, as it captures frequencies up to 8 kHz, covering most of the important information in human speech. For music, a higher sampling rate like **44.1 kHz** is standard, capturing frequencies up to 22.05 kHz, the upper limit of human hearing.

Libraries like [librosa](https://librosa.org/doc/latest/index.html) and [pydub](https://github.com/jiaaro/pydub) simplify tasks such as generating spectrograms, adjusting sampling rates, and feature extraction, aiding robust speech processing.

<img src="./../images/melSpec.png" alt="melSpec" style="clear: both; max-width: 100%; height: auto;">

*Comparison of Mel spectrograms for an audio clip saying "Hello World" at 16 kHz and 44.1 kHz sampling rates. Higher sampling rates capture more detailed information, as seen in the richer frequency representation at 44.1 kHz.*

## 1.3 Speech Applications
<a id="speech-applications"></a>
Speech technology encompasses a broad range of applications that facilitate various tasks. Among these, Automatic Speech Recognition and Speech Synthesis are two of the most influential and widely used.

**Automatic Speech Recognition (ASR)**, also known as **Speech-to-Text (SST)**, converts human speech into written text. ASR at the foundation of numerous technologies, such as virtual assistants, transcription services, real-time translation systems, and accessibility tools. To build a reliable ASR system, large datasets of transcribed speech, covering diverse speech patterns, accents, and languages, are essential. Its performance is typically evaluated using metrics like **Word Error Rate (WER)** or **Character Error Rate (CER)** on datasets such as [FLEURS](https://huggingface.co/datasets/google/fleurs) (a widely recognized benchmark for multilingual ASR, covering 102 languages)<a href="https://huggingface.co/learn/audio-course/en/chapter1/audio_data" style="text-decoration: none;" target="_blank" title="Check out this HuggingFace course to learn more"><sup>4</sup></a>.

**Speech Synthesis**, also known as **Text-to-Speech (TTS)**, converts written text into natural-sounding speech.  It is used in virtual assistants, audiobooks, language learning platforms, and tools for visually impaired users. High-quality TTS models typically need labeled audio data from a single speaker to produce realistic voices. Evaluation metrics include subjective measures like **Mean Opinion Score (MOS)** and objective metrics like **Word Error Rate (WER)** or **Character Error Rate (CER)** to assess the naturalness and accuracy of synthesized speech.

Some other common speech technology tasks include:
* **Audio Classification**: Categorizing audio clips into distinct classes.
* **Language Identification**: Identifying the language spoken in an audio clip.
* **Speaker Diarization**: Identifying the speaker at any given moment in an audio clip.
* **Keyword Detection**: Detecting specific words or phrases in continuous speech, often used for wake-word detection in voice assistants.

## 1.4 Notable Architectures
<a id="notable-architectures"></a>
There have been various architectures proposed for various speech tasks throughout the years. Here I would discuss 2 popular types. 

### Traditional Approach: Hidden Markov Models (HMM)
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

**CTC (Connectionist Temporal Classification)**: Encoder-only structure<a href="https://distill.pub/2017/ctc/" style="text-decoration: none;" target="_blank" title="Check out this tutorial to learn more"><sup>5</sup></a>
1. **Encoder**: The model processes the audio, breaking it into small frames (e.g., 20 milliseconds) and creates a hidden state for each frame.
2. **Vocabulary**: For each hidden state, the model predicts a label, such as a character or phoneme. (e.g., “HHEEE”, is predicted from the first 5 frames)
3. **Blank Token**: Since the model doesn’t know the exact timing of words, it uses a blank token (_) to skip over uncertain regions of the audio.
4. **Post-processing**: After the model makes predictions, repeated labels and blanks are cleaned up (e.g., “HHEEE_ELLL_OO” becomes “HELLO”).

Popular CTC-based ASR Models: [Wav2Vec](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2), [HuBERT](https://huggingface.co/docs/transformers/en/model_doc/hubert), [M-CTC-T](https://huggingface.co/docs/transformers/en/model_doc/mctct).

<video src="./../images/ctc.mov" autoplay loop muted style="max-width: 100%; height: auto;"></video>

*Demo of how CTC algorithm recognizes "hello" from input audio by [Awni Hannun](https://distill.pub/2017/ctc/)*

**Seq2Seq (Sequence-to-Sequence)**: Encoder-Decoder structure<a href="https://huggingface.co/learn/audio-course/en/chapter3/seq2seq" style="text-decoration: none;" target="_blank" title="Check out this tutorial to learn more"><sup>6</sup></a>
1. **Encoder**: The model listens to an audio clip (e.g., “hello”) and transforms it into a sequence of features (e.g., spectrogram). The encoder processes these features to capture key information about the speech.
2. **Decoder**: The decoder uses the encoded information to generate the output sequence. It starts with a “start” token and predicts the next word or character (e.g., “H” for “hello”), adding each prediction to the growing output.
3. **Cross-Attention**: The decoder uses the encoder’s hidden states to focus on relevant parts of the input when predicting subsequent tokens.
4. **Autoregressive Generation**: The decoder generates one token at a time, using the previous token to predict the next. For example, after predicting “H,” it uses that to predict “E,” and so on, until it finishes with ‘HELLO’

Popular Seq2Seq-based ASR models: [Whisper](https://openai.com/index/whisper/)

Popular Seq2Seq-based TTS models: [SpeechT5](https://github.com/microsoft/SpeechT5), [Tacotron](https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/)

<img src="./../images/seq2seq.png" alt="seq2seq" style="clear: both; max-width: 100%; height: auto;">

*The encoder-decoder structure shown in [OpenAI Whisper blog](https://openai.com/index/whisper/)*

# 2. Low-Resource Language Speech Research
<a id="low-resource-language-speech-research"></a>

**Low-resource languages** are languages with limited linguistic data resources, such as annotated text corpora or speech data, which are essential for building robust speech and language models. In recent years, efforts by various research groups and organizations have led to some significant progress in data creation and model development for low-resource languages.

### Common Voice (2017)
Common Voice is a crowdsourcing project initiated by Mozilla to create a diverse **open-source dataset** of human voice recordings. It encourages volunteers to contribute their voices in various languages, including many low-resource ones. Anyone can record themselves reading texts or validate recordings submitted by others in available languages. It also allows individuals to request new languages and assist in making them available. This community-driven effort has been crucial in expanding and diversifying the dataset. 

The project is continually expanding, with new releases every three months. The latest release, **Common Voice Corpus 20.0**, includes over **22,000 hours** of validated recordings across **130+ languages**<a href="https://commonvoice.mozilla.org" style="text-decoration: none;" target="_blank" title="Visit Mozilla Common Voice for more info"><sup>7</sup></a>.

### CMU Wilderness (2019) 

The CMU Wilderness Multilingual Speech dataset, developed by Carnegie Mellon University, provides aligned sentences and audio for over **700 languages**, focusing primarily on low-resource languages. The dataset was created by extracting audio from **New Testament** recordings and includes on average **20 hours of data per language**. Importantly, the data creation algorithms were **open-sourced**, enabling others to recreate the same datasets<a href="https://github.com/festvox/datasets-CMU_Wilderness" style="text-decoration: none;" target="_blank" title="Visit CMU Wilderness Github Repo for more info"><sup>8</sup></a>. This extensive dataset has enabled researchers to investigate the effectiveness of various pre-trained speech models on low-resource languages. 

### ASR2K (2022)

The ASR2K project, also from Carnegie Mellon University, aimed to improve **automatic speech recognition** for **1,909 languages** by leveraging transfer learning and multilingual models. Their approach involves mapping the output of multilingual models trained on high-resource languages to the appropriate phonemes for the target language. ASR2K represents the first attempt to build a speech recognition pipeline for thousands of languages without requiring audio<a href="https://arxiv.org/abs/2209.02842" style="text-decoration: none;" target="_blank" title="Check out the ASR2K paper to learn more"><sup>9</sup></a>.

### Meta MMS (2023)

In 2023, Meta introduced its Massively Multilingual Speech (MMS) project, which extended the capabilities of **automatic speech recognition** and **text-to-speech** systems to over **1,100 languages** and **language identification** to over **4,000 languages**.

The project utilized **Wav2Vec 2.0 models (CTC-based)** trained on datasets created using self-supervised learning and data alignment techniques. These datasets include a labeled dataset (MMS-lab) covering 1,107 languages and an unlabeled dataset (MMS-unlab) covering 3,809 languages, both created from recordings of New Testaments similar to the CMU Wilderness dataset. Meta also **open-sourced** their data alignment algorithms, enabling others to replicate and build upon their work<a href="https://github.com/facebookresearch/fairseq/tree/main/examples/mms" style="text-decoration: none;" target="_blank" title="Visit MMS Github Repo for more info"><sup>10</sup></a>.

Meta-MMS models outperformed some existing state-of-the-art models, including OpenAI's Whisper, with half the word error rate while covering 11 times more languages<a href="https://ai.meta.com/blog/multilingual-model-speech-recognition/" style="text-decoration: none;" target="_blank" title="Check out the MMS blog to learn more"><sup>11</sup></a>. These models are not only some of the most comprehensive and high-performing in the field but are also open-sourced, allowing researchers and developers to fine-tune them for specific applications, paving the way for more inclusive and accessible speech technologies worldwide. 

To see examples of fine-tuning these models for specific low-resource languages, continue to the next section.


# 3. Uyghur Speech Technology
<a id="uyghur-speech-technology"></a>

Uyghur, spoken by over 13 million people worldwide—primarily in Central Asia—represents a rich cultural and linguistic heritage. It is closely related to Uzbek, both belonging to the **Karluk** branch of the **Turkic language family**, and shares significant similarities with other Turkic languages like Kazakh, Turkish, etc.

Modern Uyghur has a unique alphabet with **32 letters**, including **8 vowels** and **24 consonants**. The language is primarily written using a modified **Perso-Arabic** script that marks all vowels, setting it apart from other languages that use similar scripts. Although less common, **Latin** and **Cyrillic** scripts are also used in various contexts. Like many Turkic languages, Uyghur features vowel harmony and agglutination and lacks noun classes or grammatical gender<a href="https://en.wikipedia.org/wiki/Uyghur_language" style="text-decoration: none;" target="_blank" title="Read more about Uyghur language here"><sup>12</sup></a>.

Over the years, substantial progress has been made in advancing Uyghur speech technology, driven by the efforts of researchers, organizations, and individuals. This overview will provide a brief history of these developments, including references for readers interested in learning more, and present available datasets, models, implementation tutorials, and tools for those looking to contribute to this field. The ultimate goal is to inspire the creation of more applications aimed at preserving the Uyghur language and potentially provide insights that can benefit other low-resource languages.


## 3.1 Brief History of Uyghur Language and Speech Technology
<a id="brief-history-of-uyghur-language-and-speech-technology"></a>

* **1994**: The Uyghur alphabet was incorporated into the international Unicode encoding standard[<sup>13</sup>](#13).
* **1996**: The first Uyghur speech corpus was created, marking the start of Uyghur speech recognition research[<sup>14</sup>](#14).
* **1998**: The first Uyghur website was established[<sup>13</sup>](#13).
* **2004**: The first Uyghur search engine was developed[<sup>15</sup>](#15).
* **2009**: A Uyghur continuous speech recognition system based on Hidden Markov Models (HMM) was developed[<sup>16</sup>](#16).
* **2014**:  
    * Detailed acoustic feature analysis of Uyghur phonemes was conducted[<sup>17</sup>](#17).  
    * A study on Uyghur speech retrieval was published[<sup>18</sup>](#18).
* **2015**: Deep neural networks began to be applied to Uyghur speech recognition[<sup>19, </sup>](#19)[<sup>20</sup>](#20).
* **2016**: Research focused on Uyghur speaker adaptation for speech recognition[<sup>21</sup>](#21).
* **2017**: The THUYG20, a free Uyghur speech corpus, was released[<sup>22</sup>](#22).
* **2021**: The state-of-the-art Conformer-CTC Uyghur speech recognition model was introduced[<sup>23</sup>](#23).
* **2022**: A comprehensive overview of speech recognition techniques for Uyghur, Kazakh, and Kyrgyz was published[<sup>24</sup>](#24).
* **2023**:  
    * Nazarbayev University released Turkic TTS, including a TTS model for Uyghur[<sup>25</sup>](#25).  
    * Meta released its MMS models, featuring ASR, TTS, and language identification for Uyghur[<sup>26</sup>](#26).

## 3.2 Open Source Uyghur Speech Datasets
<a id="open-source-uyghur-speech-datasets"></a>

## 3.3 Open Source Uyghur Speech Models
<a id="open-source-uyghur-speech-models"></a>

## 3.4 Additional Tools 
<a id="additional-tools"></a>

# 4. Reference
<a id="references"></a>

<a id="1"></a>1. Ethnologue. *Languages of the World*.

<a id="2"></a>2. VoiceScienceWorks. *Harmonics vs Formants*.

<a id="3"></a>3. Hugging Face. *Audio Course: Chapter 1 - Audio Data*.

<a id="4"></a>4. Kingma, D. P., & Ba, J. (2022). *Training Transformers with Causal Language Modeling*. arXiv:2205.1244.

<a id="5"></a>5. Graves, A. (2017). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*. Distill.

<a id="6"></a>6. Hugging Face. *Audio Course: Chapter 3 - Sequence to Sequence Models*.

<a id="7"></a>7. Mozilla. *Common Voice: Open Source Speech Dataset*.

<a id="8"></a>8. CMU Wilderness. *CMU Wilderness Dataset*.

<a id="9"></a>9. Bai, Q., Wang, Y., & Wang, X. (2022). *Scaling Multilingual Speech Recognition with MMS*. arXiv:2209.02842.

<a id="10"></a>10. Facebook Research. *Fairseq: Multilingual Speech Models*.

<a id="11"></a>11. Meta AI. (2022). *Multilingual Model for Speech Recognition*.

<a id="12"></a>12. Wikipedia Contributors. (2023). *Uyghur Language*. Wikipedia.

<a id="13"></a>13. Xu, G., Nuerguli, Liu, Q., Yang, M., Feng, G., & Wu, S. (2015). Summary of the Uyghur language information processing. *International Conference on Advances in Mechanical Engineering and Industrial Informatics (AMEII 2015)*.

<a id="14"></a>14. Wang, K., Fan, Z., Turhunjan, Fang, X., Xu, S., & Wumaier. (1998). Integrated speech corpus system of Uyghur language. *In Proceedings of the NCMMSC, Harbin, China, 30 July 1998; pp. 366–368.*

<a id="15"></a>15. Aisaitijiang, A. (2004). The Design of Uyghur Search Engine and Its Carrying-Out. *Journal of Xinjiang Education Institute*, 20(4), 102–106.

<a id="16"></a>16. Silamu, W., Tursun, N., & Saltiniyaz, P. (2009). Speech processing technology of Uyghur language. *In Proceedings of the 2009 Oriental COCOSDA International Conference on Speech Database and Assessments, Urumqi, China, 10–12 August 2009; pp. 19–24.*

<a id="17"></a>17. Wang, H., Nurmemet, Y., & Wushouer, S. (2014). Acoustic feature analysis of the Uyghur phonemes. *In Proceedings of the Faculty of Information Science and Engineering, Xinjiang University; Key Laboratory of Multilingual Information Technology, Urumqi, Xinjiang, China.*

<a id="18"></a>18. Zhang, L., Nurmemet, Y., & Wushouer, S. (2014). Study on Uyghur speech retrieval. *In Proceedings of the College of Information Science and Engineering, Xinjiang University, Urumqi, Xinjiang, China.*

<a id="19"></a>19. Tuerxun, T., & Dai, L. (2015). Deep neural network-based Uyghur large vocabulary continuous speech recognition. *Journal of Data Acquisition and Processing*, 30, 365–371.

<a id="20"></a>20. Batexi, Q., Huanag, H., & Wang, X. (2015). Uyghur speech recognition based on deep neural networks. *Computer Engineering and Design*, 36, 2239–2244.

<a id="21"></a>21. Nurmemet, Y., Zhang, L., & Wushouer, S. (2016). Speaker adaptation technology in Uyghur continuous speech recognition. *In Proceedings of the College of Information Science and Engineering, Xinjiang University, Urumqi, Xinjiang, China.*

<a id="22"></a>22. Rouzi, A., Shi, Y., Zhiyong, Z., Dong, W., Hamdulla, A., & Fang, Z. (2017). THUYG-20: A free Uyghur speech database. *Journal of Tsinghua University Science and Technology*, 57, 182–187.

<a id="23"></a>23. Subi, A., Nurmemet, Y., Huang, H., & Wushouer, S. (2021). End-to-end Uyghur speech recognition based on multi-task learning. *Journal of Signal Processing*, 37, 1852–1859.

<a id="24"></a>24. Du, W., Maimaitiyiming, Y., Nijat, M., Li, L., Hamdulla, A., & Wang, D. (2023). Automatic Speech Recognition for Uyghur, Kazakh, and Kyrgyz: An Overview. *Applied Sciences*, 13, 326.

<a id="25"></a>25. Yeshpanov, R., Mussakhojayeva, S., & Khassanov, Y. (2023). Multilingual text-to-speech synthesis for Turkic languages using transliteration. *Institute of Smart Systems and Artificial Intelligence, Nazarbayev University, Kazakhstan.*

<a id="26"></a>26. Pratap, V., Tjandra, A., Shi, B., Tomasello, P., Babu, A., Kundu, S., Elkahky, A., Ni, Z., Vyas, A., Fazel-Zarandi, M., Baevski, A., Adi, Y., Zhang, X., Hsu, W.-N., Conneau, A., & Auli, M. (2023). Scaling Speech Technology to 1,000+ Languages. *arXiv:2305.13516 [cs.CL]*.