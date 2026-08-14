# LLMs Get Lost in Evolving User Intent

## OVERVIEW

LLMs Get Lost in Evolving User Intent is a research framework and accompanying code that transforms static, single-turn benchmarks into dynamic, multi-turn conversations in which a simulated user's intent evolves across turns—incrementally revealed, revised, and at times redirected mid-conversation—while preserving each task's original, automatically verifiable evaluation protocol.

**Note:** This is a code-only release. No model or dataset is released as part of this asset, so there is no associated model card or dataset card. The framework operates on existing third-party single-turn benchmarks (see Evaluation).

## WHAT CAN LLMs Get Lost in Evolving User Intent DO

LLMs Get Lost in Evolving User Intent was developed to study how well LLM-based collaborative agents track and act on user intent as it evolves over the course of a conversation. It converts any verifiable single-turn dataset into a multi-turn environment with evolving user intent, while preserving automatic verifiability against the original answer, by modeling three forms of conversational dynamics: argument reveal (incremental disclosure), argument revision (changing a previously stated value), and function switch (pivoting to a related task).

A detailed discussion of LLMs Get Lost in Evolving User Intent, including how it was developed and tested, can be found in our paper at: https://arxiv.org/abs/2607.20734

## INTENDED USES

LLMs Get Lost in Evolving User Intent is best suited for researchers who want to study multi-turn LLM agent behavior and intent-tracking, or to reuse existing single-turn benchmarks as controlled, long-horizon multi-turn testbeds without new annotation.

LLMs Get Lost in Evolving User Intent is being shared with the research community to facilitate reproduction of our results and foster further research in this area.

LLMs Get Lost in Evolving User Intent is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them.

## OUT-OF-SCOPE USES

LLMs Get Lost in Evolving User Intent is not well suited for production deployment, real-world user-facing systems, or as a certification of any model's safety, quality, or fitness for a particular use.

We do not recommend using LLMs Get Lost in Evolving User Intent in commercial or real-world applications without further testing and development. It is being released for research purposes.

LLMs Get Lost in Evolving User Intent was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use.

Without further testing and development, LLMs Get Lost in Evolving User Intent should not be used in sensitive domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities.

We do not recommend using LLMs Get Lost in Evolving User Intent in the context of high-risk decision making (e.g. in law enforcement, legal, finance, or healthcare).

## HOW TO GET STARTED

To begin using LLMs Get Lost in Evolving User Intent, clone the repository at https://github.com/microsoft/evolving-intent/ and follow the setup and usage instructions in the repository's README (installation, configuring the LLMs used for intent extraction / simulation, and running the evolving-intent evaluation on a supported benchmark).

## EVALUATION

LLMs Get Lost in Evolving User Intent was evaluated on its ability to convert verifiable single-turn benchmarks into dynamic multi-turn conversations with evolving user intent, and to measure LLM-agent performance under those dynamics, scored by each source dataset's native verifier.

A detailed discussion of our evaluation methods and results can be found in our paper at: https://arxiv.org/abs/2607.20734

### EVALUATION METHODS

We used each dataset's native verifier (task accuracy; exact match accuracy for math, SQL, search and solvability for software engineering) to measure performance under the evolving-intent setting.

We compared the performance of LLM agents in the evolving-intent (multi-turn) setting against their fully-specified single-turn baselines using four verifiable benchmarks: GSM8K (math), BIRD-SQL (text-to-SQL), BrowseComp+ (agentic search), and SWE-Bench Verified (software engineering).

The models used for evaluation were [GPT-5.1](https://developers.openai.com/api/docs/models/gpt-5.1), [GPT-5.2](https://developers.openai.com/api/docs/models/gpt-5.2), [GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4) (including nano and mini variants), [GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5), [Gemini 3.1 Pro](https://deepmind.google/models/model-cards/gemini-3-1-pro/), [Grok 4.20](https://docs.x.ai/developers/models/grok-4.20), [Mistral Large 3](https://docs.mistral.ai/models/model-cards/mistral-large-3-25-12), [DeepSeek V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2), [Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5), and [Kimi K2.6](https://huggingface.co/moonshotai/Kimi-K2.6).

Results may vary if LLMs Get Lost in Evolving User Intent is used with a different model based on its unique design, configuration, and training.

[high level summary review of RAI/DSB testing, if any]

We measure an LLM agent's accuracy at the final turn of a multi-turn interaction in which the user's intent evolves. For example, consider the following math conversation: "Jihoon ate 2 apples." -> "Jihoon ate 3 more apples." -> "Sorry, he only ate 2 more apples, not 3." -> "How many apples did Jihoon eat?" The correct answer is 4.

### EVALUATION RESULTS

At a high level, we found that strong single-turn performance does not transfer to the evolving-intent setting, with substantial degradation across model families. For example, GPT-5.5 dropped from 99.0% to 80.5% on the math domain (GSM8K) after just 6 intent transitions, with similar drops observed across model families and tasks.

## LIMITATIONS

LLMs Get Lost in Evolving User Intent was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios.

LLMs Get Lost in Evolving User Intent was designed and tested using the English language. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native speaker of that language.

Outputs generated by AI may include factual errors, fabrication, or speculation. Users are responsible for assessing the accuracy of generated content. All decisions leveraging outputs of the system should be made with human oversight and not be based solely on system outputs.

LLMs Get Lost in Evolving User Intent inherits any biases, errors, or omissions produced by its base model. Developers are advised to choose an appropriate base LLM/MLLM carefully, depending on the intended use case.

LLMs Get Lost in Evolving User Intent was evaluated using [GPT-5.1](https://developers.openai.com/api/docs/models/gpt-5.1), [GPT-5.2](https://developers.openai.com/api/docs/models/gpt-5.2), [GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4) (including nano and mini variants), [GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5), [Gemini 3.1 Pro](https://deepmind.google/models/model-cards/gemini-3-1-pro/), [Grok 4.20](https://docs.x.ai/developers/models/grok-4.20), [Mistral Large 3](https://docs.mistral.ai/models/model-cards/mistral-large-3-25-12), [DeepSeek V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2), [Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5), and [Kimi K2.6](https://huggingface.co/moonshotai/Kimi-K2.6). See the linked documentation to understand each model's capabilities and limitations.

LLMs Get Lost in Evolving User Intent inherits any biases, errors, or omissions characteristic of its training data, which may be amplified by any AI-generated interpretations.

There has not been a systematic effort to ensure that systems using LLMs Get Lost in Evolving User Intent are protected from security vulnerabilities such as indirect prompt injection attacks. Any systems using it should take proactive measures to harden their systems as appropriate.

## BEST PRACTICES

Better performance and more reliable simulations can be achieved by using strong, well-aligned LLMs for the intent extraction, counterfactual/predecessor synthesis, and user-turn rendering steps of the pipeline.

We strongly encourage users to use LLMs/MLLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services. Such services continually update their safety and RAI mitigations with the latest industry standards for responsible use. For more on AOAI's best practices when employing foundations models for scripts and applications:

- [What is Azure AI Content Safety?](https://learn.microsoft.com/en-us/azure/ai-services/contentsafety/overview)
- [Overview of Responsible AI practices for Azure OpenAI models](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview)
- [Azure OpenAI Transparency Note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note)
- [OpenAI's Usage Policies](https://openai.com/policies/usage-policies)
- [Azure OpenAI's Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/codeofconduct)

Users are responsible for sourcing their datasets legally and ethically. This could include securing appropriate rights, ensuring consent for use of audio/images, and/or the anonymization of data prior to use in research.

Users are reminded to be mindful of data privacy concerns and are encouraged to review the privacy policies associated with any models and data storage solutions interfacing with LLMs Get Lost in Evolving User Intent.

It is the user's responsibility to ensure that the use of LLMs Get Lost in Evolving User Intent complies with relevant data protection regulations and organizational guidelines.

Developers should follow transparency best practices and inform end-users they are interacting with an AI system.

## LICENSE

MIT License

Nothing disclosed here, including the Out of Scope Uses section, should be interpreted as or deemed a restriction or modification to the license the code is released under.

## TRADEMARKS

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## CONTACT

This research was conducted by members of [Microsoft Research](https://www.microsoft.com/en-us/research/). We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact us at jihoontack@microsoft.com.

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.