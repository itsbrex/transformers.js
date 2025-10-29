

import { Pipeline, prepareImages } from './_base.js';
import { RawImage } from '../utils/image.js';

import { interpolate_4d } from '../utils/tensor.js';

/**
 * @typedef {import('./_base.js').ImagePipelineConstructorArgs} ImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImagePipelineInputs} ImagePipelineInputs
 */

/**
 * @callback ImageToImagePipelineCallback Transform the image(s) passed as inputs.
 * @param {ImagePipelineInputs} images The images to transform.
 * @returns {Promise<RawImage|RawImage[]>} The transformed image or list of images.
 *
 * @typedef {ImagePipelineConstructorArgs & ImageToImagePipelineCallback & Disposable} ImageToImagePipelineType
 */


/**
 * @typedef {Object} DepthEstimationPipelineOutput
 * @property {import('../utils/tensor.js').Tensor} predicted_depth The raw depth map predicted by the model.
 * @property {RawImage} depth The processed depth map as an image (with the same size as the input image).
 *
 * @callback DepthEstimationPipelineCallback Predicts the depth for the image(s) passed as inputs.
 * @param {ImagePipelineInputs} images The images to compute depth for.
 * @returns {Promise<DepthEstimationPipelineOutput|DepthEstimationPipelineOutput[]>} An image or a list of images containing result(s).
 *
 * @typedef {ImagePipelineConstructorArgs & DepthEstimationPipelineCallback & Disposable} DepthEstimationPipelineType
 */

/**
 * Depth estimation pipeline using any `AutoModelForDepthEstimation`. This pipeline predicts the depth of an image.
 *
 * **Example:** Depth estimation w/ `Xenova/dpt-hybrid-midas`
 * ```javascript
 * const depth_estimator = await pipeline('depth-estimation', 'Xenova/dpt-hybrid-midas');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
 * const out = await depth_estimator(url);
 * // {
 * //   predicted_depth: Tensor {
 * //     dims: [ 384, 384 ],
 * //     type: 'float32',
 * //     data: Float32Array(147456) [ 542.859130859375, 545.2833862304688, 546.1649169921875, ... ],
 * //     size: 147456
 * //   },
 * //   depth: RawImage {
 * //     data: Uint8Array(307200) [ 86, 86, 86, ... ],
 * //     width: 640,
 * //     height: 480,
 * //     channels: 1
 * //   }
 * // }
 * ```
 */
export class DepthEstimationPipeline
    extends /** @type {new (options: ImagePipelineConstructorArgs) => DepthEstimationPipelineType} */ (Pipeline)
{
    /**
     * Create a new DepthEstimationPipeline.
     * @param {ImagePipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {DepthEstimationPipelineCallback} */
    async _call(images) {
        const preparedImages = await prepareImages(images);

        const inputs = await this.processor(preparedImages);
        const { predicted_depth } = await this.model(inputs);

        const toReturn = [];
        for (let i = 0; i < preparedImages.length; ++i) {
            const batch = predicted_depth[i];
            const [height, width] = batch.dims.slice(-2);
            const [new_width, new_height] = preparedImages[i].size;

            // Interpolate to original size
            const prediction = (
                await interpolate_4d(batch.view(1, 1, height, width), {
                    size: [new_height, new_width],
                    mode: 'bilinear',
                })
            ).view(new_height, new_width);

            const minval = /** @type {number} */ (prediction.min().item());
            const maxval = /** @type {number} */ (prediction.max().item());
            const formatted = prediction
                .sub(minval)
                .div_(maxval - minval)
                .mul_(255)
                .to('uint8')
                .unsqueeze(0);
            const depth = RawImage.fromTensor(formatted);
            toReturn.push({
                predicted_depth: prediction,
                depth,
            });
        }
        return toReturn.length > 1 ? toReturn : toReturn[0];
    }
}
