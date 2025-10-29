
import { Pipeline } from './_base.js';

import { Tensor, topk } from '../utils/tensor.js';
import { softmax } from '../utils/maths.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

function isChat(x) {
    return Array.isArray(x) && x.every((x) => 'role' in x && 'content' in x);
}

/**
 * @typedef {import('../tokenizers.js').Message[]} Chat
 *
 * @typedef {Object} TextGenerationSingle
 * @property {string|Chat} generated_text The generated text.
 * @typedef {TextGenerationSingle[]} TextGenerationOutput
 *
 * @typedef {Object} TextGenerationSpecificParams Parameters specific to text-generation pipelines.
 * @property {boolean} [add_special_tokens] Whether or not to add special tokens when tokenizing the sequences.
 * @property {boolean} [return_full_text=true] If set to `false` only added text is returned, otherwise the full text is returned.
 * @typedef {import('../generation/configuration_utils.js').GenerationConfig & TextGenerationSpecificParams} TextGenerationConfig
 *
 * @callback TextGenerationPipelineCallback Complete the prompt(s) given as inputs.
 * @param {string|string[]|Chat|Chat[]} texts One or several prompts (or one list of prompts) to complete.
 * @param {Partial<TextGenerationConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<TextGenerationOutput|TextGenerationOutput[]>} An array or object containing the generated texts.
 *
 * @typedef {TextPipelineConstructorArgs & TextGenerationPipelineCallback & Disposable} TextGenerationPipelineType
 */

/**
 * Language generation pipeline using any `ModelWithLMHead` or `ModelForCausalLM`.
 * This pipeline predicts the words that will follow a specified text prompt.
 * NOTE: For the full list of generation parameters, see [`GenerationConfig`](./utils/generation#module_utils/generation.GenerationConfig).
 *
 * **Example:** Text generation with `Xenova/distilgpt2` (default settings).
 * ```javascript
 * const generator = await pipeline('text-generation', 'Xenova/distilgpt2');
 * const text = 'I enjoy walking with my cute dog,';
 * const output = await generator(text);
 * // [{ generated_text: "I enjoy walking with my cute dog, and I love to play with the other dogs." }]
 * ```
 *
 * **Example:** Text generation with `Xenova/distilgpt2` (custom settings).
 * ```javascript
 * const generator = await pipeline('text-generation', 'Xenova/distilgpt2');
 * const text = 'Once upon a time, there was';
 * const output = await generator(text, {
 *   temperature: 2,
 *   max_new_tokens: 10,
 *   repetition_penalty: 1.5,
 *   no_repeat_ngram_size: 2,
 *   num_beams: 2,
 *   num_return_sequences: 2,
 * });
 * // [{
 * //   "generated_text": "Once upon a time, there was an abundance of information about the history and activities that"
 * // }, {
 * //   "generated_text": "Once upon a time, there was an abundance of information about the most important and influential"
 * // }]
 * ```
 *
 * **Example:** Run code generation with `Xenova/codegen-350M-mono`.
 * ```javascript
 * const generator = await pipeline('text-generation', 'Xenova/codegen-350M-mono');
 * const text = 'def fib(n):';
 * const output = await generator(text, {
 *   max_new_tokens: 44,
 * });
 * // [{
 * //   generated_text: 'def fib(n):\n' +
 * //     '    if n == 0:\n' +
 * //     '        return 0\n' +
 * //     '    elif n == 1:\n' +
 * //     '        return 1\n' +
 * //     '    else:\n' +
 * //     '        return fib(n-1) + fib(n-2)\n'
 * // }]
 * ```
 */
export class TextGenerationPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => TextGenerationPipelineType} */ (Pipeline)
{
    /**
     * Create a new TextGenerationPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {TextGenerationPipelineCallback} */
    async _call(texts, generate_kwargs = {}) {
        let isBatched = false;
        let isChatInput = false;

        // By default, do not add special tokens, unless the tokenizer specifies otherwise
        let add_special_tokens =
            generate_kwargs.add_special_tokens ??
            (this.tokenizer.add_bos_token || this.tokenizer.add_eos_token) ??
            false;

        // Normalize inputs
        /** @type {string[]} */
        let inputs;
        if (typeof texts === 'string') {
            inputs = texts = [texts];
        } else if (Array.isArray(texts) && texts.every((x) => typeof x === 'string')) {
            isBatched = true;
            inputs = /** @type {string[]} */ (texts);
        } else {
            if (isChat(texts)) {
                texts = [/** @type {Chat} */ (texts)];
            } else if (Array.isArray(texts) && texts.every(isChat)) {
                isBatched = true;
            } else {
                throw new Error('Input must be a string, an array of strings, a Chat, or an array of Chats');
            }
            isChatInput = true;

            // If the input is a chat, we need to apply the chat template
            inputs = /** @type {string[]} */ (
                /** @type {Chat[]} */ (texts).map((x) =>
                    this.tokenizer.apply_chat_template(x, {
                        tokenize: false,
                        add_generation_prompt: true,
                    }),
                )
            );
            add_special_tokens = false; // Chat template handles this already
        }

        // By default, return full text
        const return_full_text = isChatInput ? false : (generate_kwargs.return_full_text ?? true);

        this.tokenizer.padding_side = 'left';
        const text_inputs = this.tokenizer(inputs, {
            add_special_tokens,
            padding: true,
            truncation: true,
        });

        const outputTokenIds = /** @type {Tensor} */ (
            await this.model.generate({
                ...text_inputs,
                ...generate_kwargs,
            })
        );

        const decoded = this.tokenizer.batch_decode(outputTokenIds, {
            skip_special_tokens: true,
        });

        let promptLengths;
        if (!return_full_text && text_inputs.input_ids.dims.at(-1) > 0) {
            promptLengths = this.tokenizer
                .batch_decode(text_inputs.input_ids, {
                    skip_special_tokens: true,
                })
                .map((x) => x.length);
        }

        /** @type {TextGenerationOutput[]} */
        const toReturn = Array.from({ length: texts.length }, (_) => []);
        for (let i = 0; i < decoded.length; ++i) {
            const textIndex = Math.floor((i / outputTokenIds.dims[0]) * texts.length);

            if (promptLengths) {
                // Trim the decoded text to only include the generated part
                decoded[i] = decoded[i].slice(promptLengths[textIndex]);
            }
            toReturn[textIndex].push({
                generated_text: isChatInput
                    ? [.../** @type {Chat[]} */ (texts)[textIndex], { role: 'assistant', content: decoded[i] }]
                    : decoded[i],
            });
        }
        return !isBatched && toReturn.length === 1 ? toReturn[0] : toReturn;
    }
}
