import { Pipeline } from './_base.js';

import { Tensor, mean_pooling, quantize_embeddings } from '../utils/tensor.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * @typedef {Object} FeatureExtractionPipelineOptions Parameters specific to feature extraction pipelines.
 * @property {'none'|'mean'|'cls'|'first_token'|'eos'|'last_token'} [pooling="none"] The pooling method to use.
 * @property {boolean} [normalize=false] Whether or not to normalize the embeddings in the last dimension.
 * @property {boolean} [quantize=false] Whether or not to quantize the embeddings.
 * @property {'binary'|'ubinary'} [precision='binary'] The precision to use for quantization.
 *
 * @callback FeatureExtractionPipelineCallback Extract the features of the input(s).
 * @param {string|string[]} texts One or several texts (or one list of texts) to get the features of.
 * @param {FeatureExtractionPipelineOptions} [options] The options to use for feature extraction.
 * @returns {Promise<Tensor>} The features computed by the model.
 *
 * @typedef {TextPipelineConstructorArgs & FeatureExtractionPipelineCallback & Disposable} FeatureExtractionPipelineType
 */

/**
 * Feature extraction pipeline using no model head. This pipeline extracts the hidden
 * states from the base transformer, which can be used as features in downstream tasks.
 *
 * **Example:** Run feature extraction with `bert-base-uncased` (without pooling/normalization).
 * ```javascript
 * const extractor = await pipeline('feature-extraction', 'Xenova/bert-base-uncased', { revision: 'default' });
 * const output = await extractor('This is a simple test.');
 * // Tensor {
 * //   type: 'float32',
 * //   data: Float32Array [0.05939924716949463, 0.021655935794115067, ...],
 * //   dims: [1, 8, 768]
 * // }
 * ```
 *
 * **Example:** Run feature extraction with `bert-base-uncased` (with pooling/normalization).
 * ```javascript
 * const extractor = await pipeline('feature-extraction', 'Xenova/bert-base-uncased', { revision: 'default' });
 * const output = await extractor('This is a simple test.', { pooling: 'mean', normalize: true });
 * // Tensor {
 * //   type: 'float32',
 * //   data: Float32Array [0.03373778983950615, -0.010106077417731285, ...],
 * //   dims: [1, 768]
 * // }
 * ```
 *
 * **Example:** Calculating embeddings with `sentence-transformers` models.
 * ```javascript
 * const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
 * const output = await extractor('This is a simple test.', { pooling: 'mean', normalize: true });
 * // Tensor {
 * //   type: 'float32',
 * //   data: Float32Array [0.09094982594251633, -0.014774246141314507, ...],
 * //   dims: [1, 384]
 * // }
 * ```
 * **Example:** Calculating binary embeddings with `sentence-transformers` models.
 * ```javascript
 * const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
 * const output = await extractor('This is a simple test.', { pooling: 'mean', quantize: true, precision: 'binary' });
 * // Tensor {
 * //   type: 'int8',
 * //   data: Int8Array [49, 108, 24, ...],
 * //   dims: [1, 48]
 * // }
 * ```
 */
export class FeatureExtractionPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => FeatureExtractionPipelineType} */ (Pipeline)
{
    /**
     * Create a new FeatureExtractionPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {FeatureExtractionPipelineCallback} */
    async _call(
        texts,
        {
            pooling = /** @type {'none'} */ ('none'),
            normalize = false,
            quantize = false,
            precision = /** @type {'binary'} */ ('binary'),
        } = {},
    ) {
        // Run tokenization
        const model_inputs = this.tokenizer(texts, {
            padding: true,
            truncation: true,
        });

        // Run model
        const outputs = await this.model(model_inputs);

        // TODO: Provide warning to the user that they might be using model which was not exported
        // specifically for feature extraction
        // console.log(this.model.config)
        // console.log(outputs)

        /** @type {Tensor} */
        let result = outputs.last_hidden_state ?? outputs.logits ?? outputs.token_embeddings;

        switch (pooling) {
            case 'none':
                // Skip pooling
                break;
            case 'mean':
                result = mean_pooling(result, model_inputs.attention_mask);
                break;
            case 'first_token':
            case 'cls':
                result = result.slice(null, 0);
                break;
            case 'last_token':
            case 'eos':
                result = result.slice(null, -1);
                break;
            default:
                throw Error(`Pooling method '${pooling}' not supported.`);
        }

        if (normalize) {
            result = result.normalize(2, -1);
        }

        if (quantize) {
            result = quantize_embeddings(result, precision);
        }

        return result;
    }
}
