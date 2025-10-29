import { Pipeline, prepareImages } from './_base.js';

import { Tensor } from '../utils/tensor.js';

/**
 * @typedef {import('./_base.js').ImagePipelineConstructorArgs} ImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImagePipelineInputs} ImagePipelineInputs
 */

/**
 * @typedef {Object} ImageFeatureExtractionPipelineOptions Parameters specific to image feature extraction pipelines.
 * @property {boolean} [pool=null] Whether or not to return the pooled output. If set to `false`, the model will return the raw hidden states.
 *
 * @callback ImageFeatureExtractionPipelineCallback Extract the features of the input(s).
 * @param {ImagePipelineInputs} images One or several images (or one list of images) to get the features of.
 * @param {ImageFeatureExtractionPipelineOptions} [options] The options to use for image feature extraction.
 * @returns {Promise<Tensor>} The image features computed by the model.
 *
 * @typedef {ImagePipelineConstructorArgs & ImageFeatureExtractionPipelineCallback & Disposable} ImageFeatureExtractionPipelineType
 */

/**
 * Image feature extraction pipeline using no model head. This pipeline extracts the hidden
 * states from the base transformer, which can be used as features in downstream tasks.
 *
 * **Example:** Perform image feature extraction with `Xenova/vit-base-patch16-224-in21k`.
 * ```javascript
 * const image_feature_extractor = await pipeline('image-feature-extraction', 'Xenova/vit-base-patch16-224-in21k');
 * const url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png';
 * const features = await image_feature_extractor(url);
 * // Tensor {
 * //   dims: [ 1, 197, 768 ],
 * //   type: 'float32',
 * //   data: Float32Array(151296) [ ... ],
 * //   size: 151296
 * // }
 * ```
 *
 * **Example:** Compute image embeddings with `Xenova/clip-vit-base-patch32`.
 * ```javascript
 * const image_feature_extractor = await pipeline('image-feature-extraction', 'Xenova/clip-vit-base-patch32');
 * const url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png';
 * const features = await image_feature_extractor(url);
 * // Tensor {
 * //   dims: [ 1, 512 ],
 * //   type: 'float32',
 * //   data: Float32Array(512) [ ... ],
 * //   size: 512
 * // }
 * ```
 */
export class ImageFeatureExtractionPipeline
    extends /** @type {new (options: ImagePipelineConstructorArgs) => ImageFeatureExtractionPipelineType} */ (Pipeline)
{
    /**
     * Create a new ImageFeatureExtractionPipeline.
     * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {ImageFeatureExtractionPipelineCallback} */
    async _call(images, { pool = null } = {}) {
        const preparedImages = await prepareImages(images);
        const { pixel_values } = await this.processor(preparedImages);
        const outputs = await this.model({ pixel_values });

        /** @type {Tensor} */
        let result;
        if (pool) {
            if (!('pooler_output' in outputs)) {
                throw Error(
                    `No pooled output was returned. Make sure the model has a 'pooler' layer when using the 'pool' option.`,
                );
            }
            result = outputs.pooler_output;
        } else {
            result = outputs.last_hidden_state ?? outputs.logits ?? outputs.image_embeds;
        }
        return result;
    }
}
